"""Streaming speech generation — audio starts playing before Thinker finishes.

Three levels of streaming:
  1. StreamingTTSPipeline: streams speech tokens → audio chunks for a single text
  2. StreamingConnectorPipeline: think(full) → stream audio (no overlap)
  3. OverlappedStreamingPipeline: Thinker streams text tokens while TTS streams
     audio — true pipelining. First audio in ~0.3s instead of ~1.5s.

The T3 token-level streaming:
  1. T3 generates speech tokens one at a time (~130 tokens/s)
  2. Every CHUNK_SIZE tokens, we run S3Gen (flow → vocoder) on the chunk
  3. Each chunk produces ~0.3-0.5s of audio that can be played immediately
"""

import torch
import torch.nn.functional as F
import numpy as np
import re
import threading
from typing import Generator, Optional
from dataclasses import dataclass, field

from chatterbox.models.s3gen.const import S3GEN_SIL


@dataclass
class AudioChunk:
    """A chunk of generated audio."""
    audio: np.ndarray       # float32 waveform
    sample_rate: int
    is_final: bool          # True if this is the last chunk
    token_count: int        # how many speech tokens produced this chunk
    total_tokens: int       # cumulative tokens so far


class StreamingTTSPipeline:
    """Streaming text-to-speech using Chatterbox Turbo.

    Generates audio chunks as speech tokens are produced,
    without waiting for the full sequence.

    Usage:
        pipeline = StreamingTTSPipeline.from_pretrained(device="cuda")
        pipeline.set_voice("speaker.wav")

        for chunk in pipeline.stream("Hello, how are you today?"):
            play_audio(chunk.audio, chunk.sample_rate)  # play immediately
    """

    # How many speech tokens to buffer before running the vocoder.
    # ~50 tokens ≈ 0.4s of audio. Lower = less latency, higher = better quality.
    CHUNK_SIZE = 50

    def __init__(self, tts, device="cuda"):
        self.tts = tts
        self.device = device
        self._conds = None

    @classmethod
    def from_pretrained(cls, device="cuda"):
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        tts = ChatterboxTurboTTS.from_pretrained(device=device)
        return cls(tts, device)

    def set_voice(self, audio_prompt_path: str, exaggeration: float = 0.0):
        """Pre-compute voice conditioning from reference audio."""
        self.tts.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        self._conds = self.tts.conds

    @torch.inference_mode()
    def stream(
        self,
        text: str,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        chunk_size: int = None,
    ) -> Generator[AudioChunk, None, None]:
        """Stream audio chunks as speech tokens are generated.

        Yields AudioChunk objects that can be played immediately.
        """
        from chatterbox.tts_turbo import punc_norm
        from transformers.generation.logits_process import (
            LogitsProcessorList,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
            RepetitionPenaltyLogitsProcessor,
        )

        chunk_size = chunk_size or self.CHUNK_SIZE
        assert self._conds is not None, "Call set_voice() first"

        # Tokenize text
        text = punc_norm(text)
        text_tokens = self.tts.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        t3 = self.tts.t3
        s3gen = self.tts.s3gen

        # Prepare T3 input
        speech_start_token = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = t3.prepare_input_embeds(
            t3_cond=self._conds.t3,
            text_tokens=text_tokens,
            speech_tokens=speech_start_token,
            cfg_weight=0.0,
        )

        # Logits processors
        logits_processors = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            logits_processors.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            logits_processors.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            logits_processors.append(TopPLogitsWarper(top_p))
        if repetition_penalty != 1.0:
            logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))

        # Initial forward pass (process full context)
        llm_outputs = t3.tfmr(inputs_embeds=embeds, use_cache=True)
        hidden_states = llm_outputs[0]
        past_key_values = llm_outputs.past_key_values

        speech_hidden = hidden_states[:, -1:]
        speech_logits = t3.speech_head(speech_hidden)

        processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
        probs = F.softmax(processed_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        all_tokens = [next_token]
        chunk_tokens = [next_token]
        current_token = next_token
        total_generated = 1

        # Generation loop — accumulate tokens, vocode periodically
        max_tokens = 1000
        prev_audio_samples = 0  # track how much audio we've already yielded

        for step in range(max_tokens):
            # Check EOS
            if torch.all(current_token == t3.hp.stop_speech_token):
                break

            # Generate next token
            current_embed = t3.speech_emb(current_token)
            llm_outputs = t3.tfmr(
                inputs_embeds=current_embed,
                past_key_values=past_key_values,
                use_cache=True,
            )
            hidden_states = llm_outputs[0]
            past_key_values = llm_outputs.past_key_values
            speech_logits = t3.speech_head(hidden_states)

            input_ids = torch.cat(all_tokens, dim=1)
            processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])
            if torch.all(processed_logits == -float("inf")):
                break

            probs = F.softmax(processed_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            all_tokens.append(next_token)
            current_token = next_token
            total_generated += 1

            # Every chunk_size tokens, vocode ALL tokens so far and yield the NEW audio
            if total_generated % chunk_size == 0:
                audio, new_samples = self._vocode_all(
                    all_tokens, s3gen, prev_audio_samples, finalize=False,
                )
                if audio is not None and len(audio) > 0:
                    prev_audio_samples = new_samples
                    yield AudioChunk(
                        audio=audio,
                        sample_rate=self.tts.sr,
                        is_final=False,
                        token_count=chunk_size,
                        total_tokens=total_generated,
                    )

        # Final vocode — all tokens with silence appended
        silence = [torch.tensor([[S3GEN_SIL]], device=self.device)] * 3
        final_tokens = all_tokens + silence

        audio, new_samples = self._vocode_all(
            final_tokens, s3gen, prev_audio_samples, finalize=True,
        )
        if audio is not None and len(audio) > 0:
            yield AudioChunk(
                audio=audio,
                sample_rate=self.tts.sr,
                is_final=True,
                token_count=total_generated,
                total_tokens=total_generated,
            )

    def _vocode_all(self, token_list, s3gen, prev_samples, finalize):
        """Vocode ALL accumulated tokens, return only the NEW audio portion.

        This avoids the tensor mismatch issue with partial chunks by always
        vocoding the full sequence, then slicing out what's new.
        """
        tokens = torch.cat(token_list, dim=1)
        # Filter invalid tokens
        tokens = tokens.clamp(max=6560)
        if tokens.shape[1] == 0:
            return None, prev_samples

        try:
            wav, _ = s3gen.inference(
                speech_tokens=tokens,
                ref_dict=self._conds.gen,
                n_cfm_timesteps=2,
            )
            wav = wav.squeeze(0).cpu().numpy().astype(np.float32)
            wav = self.tts.watermarker.apply_watermark(wav, sample_rate=self.tts.sr)

            # Only yield the NEW audio (after what we've already yielded)
            new_audio = wav[prev_samples:]
            new_total = len(wav)

            if len(new_audio) == 0:
                return None, prev_samples

            return new_audio, new_total

        except Exception as e:
            print(f"  Vocode error: {e}")
            return None, prev_samples


class StreamingConnectorPipeline:
    """Full streaming pipeline: Thinker → emotion → Connector → streaming Chatterbox.

    Usage:
        pipeline = StreamingConnectorPipeline(
            lora_path="checkpoints/living-agent/lora",
            probe_ckpt="checkpoints/probe/probe_best.pt",
            voice_path="speaker.wav",
        )

        # Streaming: yields audio chunks as they're generated
        for chunk in pipeline.speak("Tell me something exciting!"):
            play_audio(chunk.audio, chunk.sample_rate)
    """

    def __init__(
        self,
        lora_path: str = "checkpoints/living-agent/lora",
        probe_ckpt: str = "checkpoints/probe/probe_best.pt",
        voice_path: str = None,
        device: str = "cuda",
        chunk_size: int = 50,
    ):
        self.device = device
        self.voice_path = voice_path
        self.chunk_size = chunk_size

        # Load Thinker + Probe
        self._load_thinker(lora_path, probe_ckpt)

        # Load streaming TTS
        self.stream_tts = StreamingTTSPipeline.from_pretrained(device=device)
        if voice_path:
            self.stream_tts.set_voice(voice_path)

    def _load_thinker(self, lora_path, probe_ckpt):
        import unsloth
        from unsloth import FastLanguageModel
        from src.model.emotion_probe import EmotionProbe
        from src.training.train_probe import HiddenStateCapture

        self.thinker, processor = FastLanguageModel.from_pretrained(
            lora_path, device_map=self.device, dtype=torch.bfloat16,
        )
        FastLanguageModel.for_inference(self.thinker)
        self.thinker.eval()
        self.tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

        hidden_size = self.thinker.config.get_text_config().hidden_size
        self.capture = HiddenStateCapture(self.thinker)

        self.probe = EmotionProbe(hidden_size=hidden_size).to(self.device)
        import os
        if os.path.exists(probe_ckpt):
            self.probe.load_state_dict(torch.load(probe_ckpt, map_location=self.device))
        self.probe.eval()

        from src.model.connector import ThinkerTalkerConnector
        self.connector = ThinkerTalkerConnector(device=self.device, use_turbo=True)

    def set_voice(self, voice_path: str):
        """Change the output voice."""
        self.voice_path = voice_path
        self.stream_tts.set_voice(voice_path)

    def think(self, user_text: str) -> dict:
        """Run Thinker: generate response + detect emotion."""
        chat = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
        tokens = self.tokenizer(chat, return_tensors="pt").to(self.device)

        with torch.no_grad():
            self.capture.clear()
            self.thinker(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_hidden_states=True,
            )

            delta_f32 = {k: v.float() for k, v in self.capture.deltanet_states.items()}
            attn_f32 = {k: v.float() for k, v in self.capture.attention_states.items()}
            probe_out = self.probe(delta_f32, attn_f32)

            gen = self.thinker.generate(
                **tokens, max_new_tokens=100, do_sample=True,
                temperature=0.7, top_p=0.9, repetition_penalty=1.1,
            )

        gen_tokens = gen[0][tokens["input_ids"].shape[1]:]
        response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        return {
            "response": response,
            "emotion_label": probe_out.get("emotion_label", "neutral"),
            "prosody": probe_out.get("prosody_labels", {}),
            "conditioning_vector": probe_out.get("conditioning_vector"),
        }

    def speak(self, user_text: str) -> Generator[AudioChunk, None, None]:
        """Full pipeline: think → clean → stream audio.

        Yields AudioChunk objects as they're generated.
        NOTE: This waits for full text before starting TTS.
        Use OverlappedStreamingPipeline for true pipelining.
        """
        # Step 1: Think
        result = self.think(user_text)

        # Step 2: Clean text + get style
        clean_text = self.connector.clean_text(result["response"])
        style = self.connector.map_style(
            result["emotion_label"],
            result["prosody"],
            result["conditioning_vector"],
        )

        print(f"  Emotion: {result['emotion_label']}")
        print(f"  Style: temp={style['temperature']:.2f}")
        print(f"  Text: \"{clean_text[:60]}...\"")

        # Step 3: Stream audio
        yield from self.stream_tts.stream(
            text=clean_text,
            temperature=style["temperature"],
            chunk_size=self.chunk_size,
        )


# ── Clause boundary detection ────────────────────────────────────────

# Sentence-ending punctuation
_SENTENCE_END = re.compile(r'[.!?](?:\s|$)')
# Clause boundaries (comma, semicolon, colon, dash) — only split if enough text
_CLAUSE_BREAK = re.compile(r'[,;:\u2014—](?:\s|$)')


def find_clause_boundary(text: str, min_chars: int = 20, max_chars: int = 150) -> Optional[int]:
    """Find a natural split point in text for TTS chunking.

    Returns the index to split at (end of clause including punctuation),
    or None if no good boundary found yet.

    Prioritizes sentence endings, falls back to clause boundaries
    if text is getting long.
    """
    # Always split at sentence endings if we have enough text
    for m in _SENTENCE_END.finditer(text):
        pos = m.end()
        if pos >= min_chars:
            return pos

    # If text is getting long, split at clause boundaries
    if len(text) > max_chars // 2:
        for m in _CLAUSE_BREAK.finditer(text):
            pos = m.end()
            if pos >= min_chars:
                return pos

    # Force split if way too long (find last space)
    if len(text) > max_chars:
        last_space = text.rfind(' ', min_chars, max_chars)
        if last_space > 0:
            return last_space + 1

    return None


# ── Overlapped Streaming Pipeline ────────────────────────────────────

class OverlappedStreamingPipeline:
    """True pipelined streaming: Thinker generates text WHILE TTS generates audio.

    Instead of:  think ALL text → then speak ALL text
    Does:        think clause1 → speak clause1 (while thinking clause2+) → speak clause2 → ...

    The Thinker streams text tokens via TextIteratorStreamer. As soon as a
    natural clause boundary is reached (sentence end, comma after enough words),
    that clause is immediately sent to the streaming TTS pipeline.

    Audio chunks are yielded as they're generated — first audio in ~0.3-0.5s
    instead of ~1.5s without overlap.

    Usage:
        pipeline = OverlappedStreamingPipeline(
            lora_path="checkpoints/living-agent/lora",
            probe_ckpt="checkpoints/probe/probe_best.pt",
            voice_path="speaker.wav",
        )

        for chunk in pipeline.stream("Tell me something exciting!"):
            play_audio(chunk.audio, chunk.sample_rate)
    """

    # Clause splitting thresholds
    MIN_CLAUSE_CHARS = 20   # don't split shorter than this
    MAX_CLAUSE_CHARS = 150  # force split if longer than this

    def __init__(
        self,
        lora_path: str = "checkpoints/living-agent/lora",
        probe_ckpt: str = "checkpoints/probe/probe_best.pt",
        voice_path: str = None,
        device: str = "cuda",
        chunk_size: int = 50,
    ):
        self.device = device
        self.voice_path = voice_path
        self.chunk_size = chunk_size

        # Load Thinker + Probe
        self._load_thinker(lora_path, probe_ckpt)

        # Load streaming TTS
        self.stream_tts = StreamingTTSPipeline.from_pretrained(device=device)
        if voice_path:
            self.stream_tts.set_voice(voice_path)

    def _load_thinker(self, lora_path, probe_ckpt):
        import unsloth
        from unsloth import FastLanguageModel
        from src.model.emotion_probe import EmotionProbe
        from src.training.train_probe import HiddenStateCapture

        self.thinker, processor = FastLanguageModel.from_pretrained(
            lora_path, device_map=self.device, dtype=torch.bfloat16,
        )
        FastLanguageModel.for_inference(self.thinker)
        self.thinker.eval()
        self.tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

        hidden_size = self.thinker.config.get_text_config().hidden_size
        self.capture = HiddenStateCapture(self.thinker)

        self.probe = EmotionProbe(hidden_size=hidden_size).to(self.device)
        import os
        if os.path.exists(probe_ckpt):
            self.probe.load_state_dict(torch.load(probe_ckpt, map_location=self.device))
        self.probe.eval()

        from src.model.connector import ThinkerTalkerConnector
        self.connector = ThinkerTalkerConnector(device=self.device, use_turbo=True)

    def set_voice(self, voice_path: str):
        """Change the output voice."""
        self.voice_path = voice_path
        self.stream_tts.set_voice(voice_path)

    def _detect_emotion(self, user_text: str) -> dict:
        """Quick forward pass for emotion detection (~50ms).

        Runs BEFORE text generation so we know the style for TTS.
        """
        chat = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
        tokens = self.tokenizer(chat, return_tensors="pt").to(self.device)

        with torch.no_grad():
            self.capture.clear()
            self.thinker(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_hidden_states=True,
            )

            delta_f32 = {k: v.float() for k, v in self.capture.deltanet_states.items()}
            attn_f32 = {k: v.float() for k, v in self.capture.attention_states.items()}
            return self.probe(delta_f32, attn_f32)

    def stream(self, user_text: str) -> Generator[AudioChunk, None, None]:
        """Overlapped streaming: Thinker text generation pipelines with TTS audio.

        1. Detect emotion (fast forward pass, ~50ms)
        2. Start Thinker text streaming in background thread
        3. Buffer text tokens, split on clause boundaries
        4. Feed each clause to streaming TTS immediately
        5. Yield audio chunks as they come

        The Thinker (0.8B, ~100 tok/s) finishes text in ~0.5s.
        First clause is ready in ~0.1-0.2s → TTS starts immediately.
        By the time clause 1 audio is done, remaining text is already buffered.
        """
        from transformers import TextIteratorStreamer

        # Step 1: Emotion detection (need this before TTS for style params)
        probe_out = self._detect_emotion(user_text)
        emotion_label = probe_out.get("emotion_label", "neutral")
        prosody = probe_out.get("prosody_labels", {})
        conditioning_vector = probe_out.get("conditioning_vector")

        style = self.connector.map_style(emotion_label, prosody, conditioning_vector)
        print(f"  Emotion: {emotion_label}")
        print(f"  Style: temp={style['temperature']:.2f}")

        # Step 2: Start Thinker text streaming in background thread
        chat = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
        tokens = self.tokenizer(chat, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_special_tokens=True, skip_prompt=True,
        )

        gen_kwargs = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "max_new_tokens": 200,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "streamer": streamer,
        }

        # Generate in background thread — streamer yields tokens as they arrive
        gen_thread = threading.Thread(target=self.thinker.generate, kwargs=gen_kwargs)
        gen_thread.start()

        # Step 3: Buffer text, split on boundaries, feed to TTS
        buffer = ""
        clause_idx = 0
        full_text = ""

        for text_piece in streamer:
            buffer += text_piece
            full_text += text_piece

            # Check for clause boundary
            boundary = find_clause_boundary(
                buffer, self.MIN_CLAUSE_CHARS, self.MAX_CLAUSE_CHARS,
            )

            if boundary is not None:
                clause = buffer[:boundary].strip()
                buffer = buffer[boundary:].lstrip()

                if clause:
                    clean = self.connector.clean_text(clause)
                    if clean:
                        clause_idx += 1
                        print(f"  Clause {clause_idx}: \"{clean[:60]}{'...' if len(clean) > 60 else ''}\"")

                        # Stream TTS for this clause — yields audio chunks
                        yield from self.stream_tts.stream(
                            text=clean,
                            temperature=style["temperature"],
                            chunk_size=self.chunk_size,
                        )

        # Step 4: Final remaining text
        remaining = buffer.strip()
        if remaining:
            clean = self.connector.clean_text(remaining)
            if clean:
                clause_idx += 1
                print(f"  Clause {clause_idx} (final): \"{clean[:60]}{'...' if len(clean) > 60 else ''}\"")

                yield from self.stream_tts.stream(
                    text=clean,
                    temperature=style["temperature"],
                    chunk_size=self.chunk_size,
                )

        gen_thread.join()
        print(f"  Full response: \"{full_text[:100]}{'...' if len(full_text) > 100 else ''}\"")
