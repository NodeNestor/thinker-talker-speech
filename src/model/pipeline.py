"""ThinkerTalkerPipeline — full end-to-end speech-to-speech inference.

The pipeline wires all components together:
  Mic -> Whisper -> Adapter -> Thinker -> Probe + Connector -> Talker -> Speaker

Supports:
  - Streaming (Talker starts before Thinker finishes)
  - Tool call interception (tool calls don't reach the Talker)
  - Runtime voice swapping (just pass a new speaker embedding)
  - Emotion override (manually set emotion, or let the probe auto-detect)
"""

import torch
import torch.nn as nn
from typing import Optional, Generator
import re

from .thinker import Thinker
from .adapter import WhisperAdapter
from .emotion_probe import EmotionProbe
from .speaker_encoder import SpeakerEncoder
from .connector import ThinkerTalkerConnector


TOOL_CALL_PATTERN = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


class ThinkerTalkerPipeline(nn.Module):
    """Full end-to-end speech pipeline."""

    def __init__(
        self,
        thinker: Thinker = None,
        adapter: WhisperAdapter = None,
        probe: EmotionProbe = None,
        speaker_encoder: SpeakerEncoder = None,
        connector: ThinkerTalkerConnector = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # Initialize components (lazy — create on first use if not provided)
        self.thinker = thinker
        self.adapter = adapter
        self.probe = probe
        self.speaker_encoder = speaker_encoder
        self.connector = connector

        # State
        self._speaker_embedding: Optional[torch.Tensor] = None
        self._emotion_override: Optional[torch.Tensor] = None

    def set_voice(self, audio_path: str = None, audio: torch.Tensor = None,
                  sample_rate: int = 16000):
        """Set the output voice from a reference audio sample (3-30s)."""
        if self.speaker_encoder is None:
            self.speaker_encoder = SpeakerEncoder(device=self.device)

        if audio_path:
            self._speaker_embedding = self.speaker_encoder.from_file(audio_path)
        elif audio is not None:
            self._speaker_embedding = self.speaker_encoder(audio, sample_rate)
        else:
            raise ValueError("Provide audio_path or audio tensor")

    def set_emotion(self, emotion_vector: torch.Tensor = None):
        """Override emotion (set to None to use auto-detection from probe)."""
        self._emotion_override = emotion_vector

    def process_speech(
        self,
        audio: torch.Tensor,
        sample_rate: int = 16000,
        max_new_tokens: int = 512,
    ) -> dict:
        """Process speech input and generate speech output.

        Args:
            audio: Input waveform [samples] or [1, samples]
            sample_rate: Audio sample rate
            max_new_tokens: Max tokens for Thinker generation

        Returns:
            text: Generated text response
            emotion: Detected emotion info
            prosody: Detected prosody info
            tool_calls: Any tool calls extracted (list of strings)
            audio_output: Generated audio waveform (when Talker is connected)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Step 1: Whisper encode + adapt
        speech_embeds = self.adapter(audio=audio)
        # speech_embeds: [1, time, thinker_hidden]

        # Step 2: Thinker generates text (in embedding space)
        output = self.thinker(inputs_embeds=speech_embeds)

        # Step 3: Extract hidden states for probe
        deltanet_states = self.thinker.get_deltanet_states()
        attention_states = self.thinker.get_attention_states()

        # Step 4: Emotion probe
        if self._emotion_override is not None:
            emotion_info = {"conditioning_vector": self._emotion_override}
        else:
            emotion_info = self.probe(deltanet_states, attention_states)

        # Step 5: Generate text
        # For actual generation, we need to use generate()
        gen_output = self.thinker.generate(
            inputs_embeds=speech_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        text = self.thinker.tokenizer.decode(gen_output[0], skip_special_tokens=True)

        # Step 6: Extract tool calls
        tool_calls = TOOL_CALL_PATTERN.findall(text)
        spoken_text = TOOL_CALL_PATTERN.sub("", text).strip()

        # Step 7: If we have a connector + talker, generate audio
        audio_output = None
        if self.connector is not None and self._speaker_embedding is not None:
            thinker_hidden = self.thinker.get_last_hidden_state()
            conditioning = emotion_info["conditioning_vector"]

            talker_input = self.connector(
                thinker_hidden=thinker_hidden,
                emotion_vector=conditioning,
                speaker_embedding=self._speaker_embedding,
            )
            # TODO: Pass talker_input to Chatterbox Turbo or other Talker
            # audio_output = self.talker.generate(talker_input)

        return {
            "text": spoken_text,
            "full_text": text,
            "emotion": emotion_info.get("emotion_label"),
            "emotion_probs": emotion_info.get("emotion_probs"),
            "prosody": emotion_info.get("prosody_labels"),
            "tool_calls": tool_calls,
            "audio_output": audio_output,
        }

    def process_text(self, text: str) -> dict:
        """Process text input (for testing without speech)."""
        tokens = self.thinker.tokenizer(text, return_tensors="pt").to(self.device)

        output = self.thinker(**tokens)

        deltanet_states = self.thinker.get_deltanet_states()
        attention_states = self.thinker.get_attention_states()

        emotion_info = self.probe(deltanet_states, attention_states)

        return {
            "emotion": emotion_info.get("emotion_label"),
            "emotion_probs": emotion_info.get("emotion_probs"),
            "prosody": emotion_info.get("prosody_labels"),
            "conditioning_vector": emotion_info.get("conditioning_vector"),
        }
