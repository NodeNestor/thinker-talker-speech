#!/usr/bin/env python3
"""Live Agent Loop — fully async pipeline.

Architecture:
  [Mic Thread]  --audio-->  [STT]  --text-->  [Input Queue]
  [Brain Thread] reads Input Queue, streams tokens, emits blocks
  [Voice Thread] reads Speech Queue, generates TTS, plays audio
  [Main Thread]  coordinates, prints live tokens

All three run concurrently:
  - Mic always listens (even while agent speaks/thinks)
  - User speaking interrupts agent speech immediately
  - TTS generates next sentence while current one plays
  - Thinking streams tokens live in gray

Usage:
    python scripts/live_agent.py                # full voice mode
    python scripts/live_agent.py --text         # keyboard input, voice output
    python scripts/live_agent.py --text --no-tts  # pure text mode
"""

import argparse
import asyncio
import json
import os
import sys
import time
import logging
import re
import threading
import queue

import torch
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.runtime.runtime import parse_speak_attrs
from src.runtime.tools import ToolExecutor

log = logging.getLogger("live_agent")

SYSTEM_PROMPT = """You are a living AI assistant running on the user's computer. You can see, hear, and act.

You respond using structured blocks:
- <think>your internal reasoning</think>  (hidden from user)
- <tool_call>{"name": "tool_name", "args": {...}}</tool_call>  (execute a tool)
- <speak emotion="neutral" speed="1.0" energy="0.7">what you say out loud</speak>

Available tools: read_file, write_file, edit_file, search_code, run_command, run_tests, git, web_search, web_fetch, check_processes, notify, set_timer, memory_store, memory_query

Guidelines:
- Always think before acting or speaking
- Use tools to interact with the computer
- Speak naturally with appropriate emotions (happy, excited, concerned, thoughtful, neutral)
- Keep spoken responses concise — you're talking, not writing an essay
- Use paralinguistic tags in speech: [laugh], [chuckle], [pause], [sigh]
- If a tool call fails, explain what happened and try an alternative
- ALWAYS wrap spoken text in <speak> tags"""


# ── Model loading ───────────────────────────────────────────────────

def load_thinker(lora_path: str, device: str = "cuda"):
    """Load Qwen 3.5 0.8B with LoRA via Unsloth (4-bit, same as training)."""
    print("Loading Thinker (Qwen 3.5 + LoRA via Unsloth)...")
    t0 = time.time()
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_path,
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    FastLanguageModel.for_inference(model)

    if hasattr(tokenizer, 'tokenizer'):
        inner_tokenizer = tokenizer.tokenizer
        print(f"  Processor detected ({type(tokenizer).__name__})")
    else:
        inner_tokenizer = tokenizer

    lora_count = sum(1 for n, _ in model.named_parameters() if 'lora' in n.lower())
    print(f"  Loaded in {time.time() - t0:.1f}s | LoRA layers: {lora_count}")
    return model, inner_tokenizer


def load_whisper(device: str = "cuda"):
    """Load faster-whisper (CTranslate2) — 4-6x faster than OpenAI whisper."""
    print("Loading faster-whisper (STT)...")
    t0 = time.time()
    from faster_whisper import WhisperModel
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel("small", device=device, compute_type=compute_type)
    print(f"  Loaded in {time.time() - t0:.1f}s ({compute_type})")
    return model


def load_chatterbox(device: str = "cuda"):
    """Load Chatterbox Turbo TTS."""
    print("Loading Chatterbox Turbo (TTS)...")
    t0 = time.time()
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    print(f"  Loaded in {time.time() - t0:.1f}s (sr={model.sr})")
    return model


# ── Audio I/O ───────────────────────────────────────────────────────

class AudioPlayer:
    """Non-blocking audio player that can be interrupted."""

    def __init__(self):
        import sounddevice as sd
        self._sd = sd
        self._playing = False
        self._lock = threading.Lock()

    def play(self, audio: np.ndarray, sr: int):
        """Play audio. Can be interrupted by calling stop()."""
        with self._lock:
            self._playing = True
        self._sd.play(audio, sr)
        self._sd.wait()
        with self._lock:
            self._playing = False

    def stop(self):
        """Stop current playback immediately."""
        with self._lock:
            if self._playing:
                self._sd.stop()
                self._playing = False

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing


class MicListener:
    """Continuous mic listener with VAD. Runs in its own thread."""

    def __init__(self, sr: int = 16000, silence_duration: float = 1.5):
        self.sr = sr
        self.silence_duration = silence_duration
        self._threshold = None
        self._stop = threading.Event()

    def calibrate(self):
        """Measure ambient noise."""
        import sounddevice as sd
        samples = int(self.sr * 0.5)
        audio = sd.rec(samples, samplerate=self.sr, channels=1, dtype="float32")
        sd.wait()
        rms = np.sqrt(np.mean(audio ** 2))
        self._threshold = max(rms * 3.0, 0.005)
        print(f"  Mic calibrated: threshold={self._threshold:.4f}")

    def record_utterance(self) -> np.ndarray:
        """Block until speech is detected, record until silence. Returns audio."""
        import sounddevice as sd

        if self._threshold is None:
            self.calibrate()

        chunk_ms = 100
        chunk_samples = int(self.sr * chunk_ms / 1000)
        silence_chunks = int(self.silence_duration / (chunk_ms / 1000))
        max_wait = int(60_000 / chunk_ms)  # Wait up to 60s
        max_record = int(30_000 / chunk_ms)  # Record up to 30s

        stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                blocksize=chunk_samples)
        stream.start()

        try:
            # Phase 1: Wait for speech
            pre_buf = []
            for _ in range(max_wait):
                if self._stop.is_set():
                    return None
                data, _ = stream.read(chunk_samples)
                pre_buf.append(data.copy())
                if len(pre_buf) > 3:
                    pre_buf.pop(0)
                if np.sqrt(np.mean(data ** 2)) > self._threshold:
                    break
            else:
                return None

            # Phase 2: Record until silence
            chunks = list(pre_buf)
            silent = 0
            for _ in range(max_record):
                if self._stop.is_set():
                    break
                data, _ = stream.read(chunk_samples)
                chunks.append(data.copy())
                if np.sqrt(np.mean(data ** 2)) > self._threshold:
                    silent = 0
                else:
                    silent += 1
                if silent >= silence_chunks:
                    break
        finally:
            stream.stop()
            stream.close()

        audio = np.concatenate(chunks, axis=0).squeeze()
        return audio

    def stop(self):
        self._stop.set()


def transcribe(whisper_model, audio: np.ndarray, sr: int = 16000) -> str:
    """Transcribe with faster-whisper."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    segments, _ = whisper_model.transcribe(audio, beam_size=1, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments).strip()


# ── TTS Pipeline ────────────────────────────────────────────────────

class TTSPipeline:
    """Async TTS: generates next sentence while current one plays."""

    def __init__(self, tts_model, player: AudioPlayer):
        self.tts = tts_model
        self.player = player
        self._interrupted = threading.Event()

    def _generate_audio(self, text: str) -> np.ndarray:
        clean = re.sub(r'<(?!/?(?:laugh|chuckle|pause|sigh|gasp))[^>]+>', '', text).strip()
        if not clean:
            return None
        wav = self.tts.generate(clean)
        audio = wav.squeeze().cpu().numpy()
        return audio[0] if audio.ndim > 1 else audio

    def speak(self, text: str):
        """Speak text with sentence-level pipelining. Can be interrupted."""
        self._interrupted.clear()

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        merged = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if merged and len(merged[-1]) < 20:
                merged[-1] += " " + s
            else:
                merged.append(s)
        if not merged:
            return

        # Pipeline: generate first, then overlap play+generate
        current_audio = self._generate_audio(merged[0])

        for i in range(len(merged)):
            if self._interrupted.is_set():
                return

            if current_audio is None:
                if i + 1 < len(merged):
                    current_audio = self._generate_audio(merged[i + 1])
                continue

            if i + 1 < len(merged):
                # Play current in background thread while generating next
                play_thread = threading.Thread(
                    target=self.player.play,
                    args=(current_audio, self.tts.sr),
                    daemon=True,
                )
                play_thread.start()

                # Generate next while playing
                if not self._interrupted.is_set():
                    next_audio = self._generate_audio(merged[i + 1])
                else:
                    self.player.stop()
                    return

                play_thread.join()
                current_audio = next_audio
            else:
                # Last sentence
                if not self._interrupted.is_set():
                    self.player.play(current_audio, self.tts.sr)

    def interrupt(self):
        """Stop speaking immediately."""
        self._interrupted.set()
        self.player.stop()


# ── Streaming LLM Generation ───────────────────────────────────────

def stream_generate(model, tokenizer, messages, max_new_tokens=2048,
                    interrupt_event=None):
    """Stream tokens from the model. Yields (token_str, accumulated_text).

    Runs generation in a background thread, yields tokens as they arrive.
    """
    from transformers import TextIteratorStreamer

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    gen_thread = threading.Thread(
        target=lambda: model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            streamer=streamer,
        ),
        daemon=True,
    )
    gen_thread.start()

    accumulated = "<think>"  # Template adds <think>\n in prompt
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]

    for chunk in streamer:
        if interrupt_event and interrupt_event.is_set():
            break

        accumulated += chunk

        # Check stop tokens
        for stop in stop_tokens:
            if stop in accumulated:
                accumulated = accumulated[:accumulated.index(stop)]
                gen_thread.join(timeout=2)
                yield chunk, accumulated
                return

        yield chunk, accumulated

    gen_thread.join(timeout=2)


def extract_blocks(text: str) -> list[dict]:
    """Extract completed blocks from accumulated text."""
    blocks = []
    pattern = re.compile(
        r"<(think|speak|tool_call)(?:\s+([^>]*))?>(.+?)</\1>",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        btype = m.group(1)
        attrs = m.group(2) or ""
        content = m.group(3).strip()

        if btype == "think":
            blocks.append({"type": "think", "content": content,
                           "start": m.start(), "end": m.end()})
        elif btype == "speak":
            pa = parse_speak_attrs(attrs)
            blocks.append({"type": "speak", "text": content,
                           "emotion": pa.get("emotion", "neutral"),
                           "speed": pa.get("speed", 1.0),
                           "energy": pa.get("energy", 0.7),
                           "start": m.start(), "end": m.end()})
        elif btype == "tool_call":
            try:
                call = json.loads(content)
                blocks.append({"type": "tool_call", "name": call["name"],
                               "args": call.get("args", {}),
                               "start": m.start(), "end": m.end()})
            except (json.JSONDecodeError, KeyError):
                pass
    return blocks


# ── The Living Agent ────────────────────────────────────────────────

class LiveAgent:
    """Fully async living agent with concurrent listening/thinking/speaking."""

    def __init__(self, thinker, tokenizer, whisper_model=None,
                 tts_model=None, workspace=".", text_mode=False, no_tts=False):
        self.thinker = thinker
        self.tokenizer = tokenizer
        self.whisper = whisper_model
        self.text_mode = text_mode
        self.no_tts = no_tts

        # Tools & memory
        self.tools = ToolExecutor(workspace=workspace)
        from src.runtime.memory import MemoryManager
        self.memory = MemoryManager(workspace=workspace)

        # Audio I/O
        self.player = AudioPlayer()
        self.tts_pipe = TTSPipeline(tts_model, self.player) if tts_model else None
        self.mic = MicListener() if not text_mode else None

        # Queues for async pipeline
        self.input_queue = queue.Queue()    # User text input
        self.interrupt = threading.Event()  # Interrupt signal

        # Conversation state
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.max_history = 50
        self._running = False
        self._speaking = False  # True while TTS is active

    def _trim_history(self):
        if len(self.messages) > self.max_history:
            self.messages = [self.messages[0]] + self.messages[-(self.max_history - 1):]

    # ── Mic listener thread ─────────────────────────────────────────

    def _mic_loop(self):
        """Continuously listen for speech. Runs in its own thread."""
        while self._running:
            try:
                sys.stdout.write("\r  \033[90m[listening...]\033[0m  ")
                sys.stdout.flush()

                audio = self.mic.record_utterance()
                if audio is None or len(audio) < 1600:  # <100ms = noise
                    continue

                duration = len(audio) / self.mic.sr
                sys.stdout.write(f"\r  [transcribing {duration:.1f}s...]   ")
                sys.stdout.flush()

                # If agent is currently speaking, interrupt it
                if self._speaking and self.tts_pipe:
                    self.tts_pipe.interrupt()
                    self.interrupt.set()

                t0 = time.time()
                text = transcribe(self.whisper, audio)
                stt_time = time.time() - t0

                if text and len(text) > 1:
                    print(f"\rYou: {text}  \033[90m({stt_time:.1f}s STT)\033[0m")
                    self.input_queue.put(text)

            except Exception as e:
                if self._running:
                    print(f"\n  [mic error: {e}]")
                time.sleep(0.5)

    # ── Brain: process input → generate → route blocks ──────────────

    def _process(self, user_text: str):
        """Process one user input through the full think→tool→speak loop."""
        self.messages.append({"role": "user", "content": user_text})
        self._trim_history()

        raw_parts = []

        for iteration in range(5):  # Max tool call chains
            if self.interrupt.is_set():
                print("  [interrupted]")
                break

            # Stream generation, print tokens live
            accumulated = ""
            in_think = True
            blocks_yielded = set()  # Track which blocks we've already processed

            sys.stdout.write("  ")
            for chunk, acc in stream_generate(
                self.thinker, self.tokenizer, self.messages,
                interrupt_event=self.interrupt,
            ):
                accumulated = acc

                # Live token printing: gray for think, white for speech
                display = chunk
                for stop in ["<|im_end|>", "<|endoftext|>"]:
                    display = display.replace(stop, "")
                if display:
                    if in_think:
                        sys.stdout.write(f"\033[90m{display}\033[0m")
                    else:
                        sys.stdout.write(display)
                    sys.stdout.flush()

                # Track think state
                if "</think>" in chunk:
                    in_think = False
                if "<think>" in chunk:
                    in_think = True

                # Check for newly completed blocks and handle speak immediately
                blocks = extract_blocks(accumulated)
                for block in blocks:
                    block_id = (block.get("start"), block.get("end"))
                    if block_id in blocks_yielded:
                        continue
                    blocks_yielded.add(block_id)

                    if block["type"] == "speak" and not self.interrupt.is_set():
                        text = block["text"]
                        emotion = block.get("emotion", "neutral")
                        raw_parts.append(f'<speak emotion="{emotion}">{text}</speak>')

                        # Start TTS in background thread immediately!
                        # This runs while the model may still be generating more tokens
                        if self.tts_pipe and not self.no_tts:
                            self._speaking = True
                            speak_thread = threading.Thread(
                                target=self._speak_and_clear,
                                args=(text,),
                                daemon=True,
                            )
                            speak_thread.start()

            print()  # Newline after generation

            # Process all blocks from final accumulated text
            blocks = extract_blocks(accumulated)
            needs_continuation = False

            for block in blocks:
                block_id = (block.get("start"), block.get("end"))

                if block["type"] == "think":
                    raw_parts.append(f'<think>{block["content"]}</think>')

                elif block["type"] == "tool_call":
                    name = block["name"]
                    args = block["args"]
                    print(f"  \033[93m[tool]\033[0m {name}({json.dumps(args)[:100]})")

                    # Route to memory or tools
                    loop = asyncio.new_event_loop()
                    if name in self.memory.memory_tools:
                        result = loop.run_until_complete(
                            self.memory.execute(name, args))
                    else:
                        result = loop.run_until_complete(
                            self.tools.execute(name, args))
                    loop.close()

                    result_str = str(result)[:2000]
                    print(f"  \033[93m[result]\033[0m {result_str[:200]}")

                    tc = f'<tool_call>{json.dumps({"name": name, "args": args})}</tool_call>'
                    tc += f'\n<tool_result>{result_str}</tool_result>'
                    self.messages.append({"role": "assistant", "content": tc})
                    raw_parts.append(tc)
                    needs_continuation = True

                elif block["type"] == "speak":
                    # Already handled during streaming above
                    pass

            # Handle leftover text (model didn't use <speak> tags)
            leftover = accumulated
            for b in blocks:
                leftover = leftover[:b["start"]] + " " * (b["end"] - b["start"]) + leftover[b["end"]:]
            leftover = re.sub(r'<[^>]+>', '', leftover).strip()
            leftover = re.sub(r'\(.*?\)', '', leftover).strip()
            if leftover and len(leftover) > 3 and any(c.isalpha() for c in leftover):
                has_speak = any(b["type"] == "speak" for b in blocks)
                if not has_speak:
                    print(f"  \033[96m[speak]\033[0m {leftover}")
                    raw_parts.append(f'<speak emotion="neutral">{leftover}</speak>')
                    if self.tts_pipe and not self.no_tts:
                        self._speaking = True
                        t = threading.Thread(
                            target=self._speak_and_clear,
                            args=(leftover,), daemon=True)
                        t.start()

            if not needs_continuation:
                self.messages.append({
                    "role": "assistant",
                    "content": "\n".join(raw_parts),
                })
                break

        # Wait for any remaining TTS to finish
        self._wait_for_speech()

    def _speak_and_clear(self, text: str):
        """Speak text and clear the speaking flag when done."""
        try:
            self.tts_pipe.speak(text)
        finally:
            self._speaking = False

    def _wait_for_speech(self):
        """Wait for TTS to finish playing."""
        while self._speaking:
            time.sleep(0.05)

    # ── Main loop ───────────────────────────────────────────────────

    def run(self):
        """Run the agent. Mic, brain, and voice all run concurrently."""
        self._running = True

        print("\n" + "=" * 60)
        print("  Living Agent")
        print("=" * 60)
        if self.text_mode:
            print("  Input:  keyboard")
        else:
            print("  Input:  microphone (always listening)")
            print("  Interrupt: just speak — agent stops and listens")
        print(f"  Output: {'text + voice' if not self.no_tts else 'text only'}")
        print(f"  Think:  \033[90mshown in gray\033[0m")
        print(f"  Tools:  \033[93mshown in yellow\033[0m")
        print("  Quit:   say 'goodbye' or Ctrl+C")
        print("=" * 60 + "\n")

        # Start mic thread
        mic_thread = None
        if not self.text_mode and self.whisper:
            self.mic.calibrate()
            mic_thread = threading.Thread(target=self._mic_loop, daemon=True)
            mic_thread.start()

        try:
            while self._running:
                try:
                    if self.text_mode:
                        user_input = input("\nYou: ").strip()
                        if not user_input:
                            continue
                    else:
                        # Block until mic picks up speech
                        user_input = self.input_queue.get()

                    # Commands
                    lower = user_input.lower().strip().rstrip(".,!?")
                    if lower in ("quit", "exit", "bye", "goodbye"):
                        print("\n  Goodbye!")
                        if self.tts_pipe and not self.no_tts:
                            self.tts_pipe.speak("Goodbye!")
                        break
                    if user_input.strip() == "/tools":
                        all_tools = self.tools.available_tools + self.memory.memory_tools
                        print(f"  Tools: {', '.join(all_tools)}")
                        continue

                    # Process
                    self.interrupt.clear()
                    t0 = time.time()
                    self._process(user_input)
                    elapsed = time.time() - t0
                    print(f"  \033[90m[{elapsed:.1f}s]\033[0m")

                    # Drain any inputs that arrived during processing
                    while not self.input_queue.empty():
                        try:
                            next_text = self.input_queue.get_nowait()
                            print(f"\nYou: {next_text}")
                            self.interrupt.clear()
                            t0 = time.time()
                            self._process(next_text)
                            print(f"  \033[90m[{time.time() - t0:.1f}s]\033[0m")
                        except queue.Empty:
                            break

                except KeyboardInterrupt:
                    print("\n\n  Ctrl+C — shutting down.")
                    break
                except Exception as e:
                    print(f"\n  \033[91m[error: {e}]\033[0m")
                    import traceback
                    traceback.print_exc()

        finally:
            self._running = False
            if self.mic:
                self.mic.stop()
            if mic_thread:
                mic_thread.join(timeout=2)


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Living Agent")
    parser.add_argument("--lora", default="checkpoints/living-agent/lora")
    parser.add_argument("--device", default=None)
    parser.add_argument("--text", action="store_true", help="Keyboard input")
    parser.add_argument("--no-tts", action="store_true", help="No voice output")
    parser.add_argument("--no-stt", action="store_true", help="No voice input")
    parser.add_argument("--workspace", default=".")
    args = parser.parse_args()

    if args.no_stt:
        args.text = True

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {args.device}")
    if args.device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print()
    thinker, tokenizer = load_thinker(args.lora, device=args.device)
    whisper_model = load_whisper(device=args.device) if not args.text else None
    tts_model = load_chatterbox(device=args.device) if not args.no_tts else None

    if args.device == "cuda":
        print(f"\nVRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    agent = LiveAgent(
        thinker=thinker, tokenizer=tokenizer,
        whisper_model=whisper_model, tts_model=tts_model,
        workspace=args.workspace, text_mode=args.text, no_tts=args.no_tts,
    )
    agent.run()


if __name__ == "__main__":
    main()
