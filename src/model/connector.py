"""Thinker->Talker Connector — translates Thinker internal state to Chatterbox style parameters.

The connector is the glue between thinking and speaking. It does NOT predict audio.
Instead, it takes:
  1. Thinker's text output (what to say)
  2. Emotion probe output (how the Thinker "feels" about it)
  3. Prosody signals (speed, pitch, energy, emphasis)

And maps them to Chatterbox's generation parameters:
  - exaggeration (emotion intensity) — full Chatterbox only
  - cfg_weight (how closely to follow conditioning)
  - temperature (sampling diversity)
  - speaking style via modified conditioning embeddings

Chatterbox does all the heavy lifting (text → speech tokens → mel → audio).
The connector just tells it HOW to say it.
"""

import re
import torch
import torch.nn as nn

from .emotion_probe import EMOTION_LABELS, PROSODY_DIMS


# Map emotion labels to base exaggeration levels
EMOTION_EXAGGERATION = {
    "neutral": 0.3,
    "happy": 0.6,
    "sad": 0.5,
    "angry": 0.8,
    "excited": 0.9,
    "fearful": 0.7,
    "surprised": 0.8,
    "disgusted": 0.6,
    "calm": 0.2,
    "confused": 0.5,
}

# Map emotion labels to temperature adjustments
EMOTION_TEMPERATURE = {
    "neutral": 0.7,
    "happy": 0.85,
    "sad": 0.75,
    "angry": 0.9,
    "excited": 0.95,
    "fearful": 0.8,
    "surprised": 0.9,
    "disgusted": 0.75,
    "calm": 0.65,
    "confused": 0.8,
}


class StyleMapper(nn.Module):
    """Learned mapping from emotion/prosody vector to Chatterbox style parameters.

    Maps the probe's 14-dim conditioning vector (10 emotions + 4 prosody) to:
      - exaggeration: float [0, 1] — emotion intensity for Chatterbox
      - cfg_weight: float [0, 1] — classifier-free guidance strength
      - temperature: float [0.5, 1.2] — sampling diversity

    Small network (~10K params) trained on (emotion_vector, target_style) pairs.
    Falls back to rule-based mapping if untrained.
    """

    def __init__(self, emotion_dim: int = 14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emotion_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 3),  # exaggeration, cfg_weight, temperature
        )

    def forward(self, conditioning_vector: torch.Tensor) -> dict:
        """
        Args:
            conditioning_vector: [batch, 14] from EmotionProbe

        Returns:
            dict with exaggeration, cfg_weight, temperature (all scalars)
        """
        raw = self.net(conditioning_vector)  # [batch, 3]
        # Constrain outputs to valid ranges
        exaggeration = torch.sigmoid(raw[:, 0])          # [0, 1]
        cfg_weight = torch.sigmoid(raw[:, 1]) * 0.8      # [0, 0.8]
        temperature = torch.sigmoid(raw[:, 2]) * 0.7 + 0.5  # [0.5, 1.2]

        return {
            "exaggeration": exaggeration.mean().item(),
            "cfg_weight": cfg_weight.mean().item(),
            "temperature": temperature.mean().item(),
        }


class ThinkerTalkerConnector:
    """Connects Thinker output to Chatterbox TTS.

    This is the pipeline glue — not a neural network predicting audio.
    It translates Thinker's internal state into Chatterbox generation calls.

    Usage:
        connector = ThinkerTalkerConnector(voice_path="speaker.wav")
        audio = connector.generate(
            text="Hello world!",
            emotion_label="happy",
            prosody={"speed": 1.2, "energy": 1.0, ...},
            conditioning_vector=probe_output,
        )
    """

    def __init__(
        self,
        voice_path: str = None,
        device: str = "cuda",
        use_turbo: bool = True,
        style_mapper: StyleMapper = None,
    ):
        self.voice_path = voice_path
        self.device = device
        self.use_turbo = use_turbo
        self.style_mapper = style_mapper
        self._tts = None
        self._tts_full = None

    @property
    def tts(self):
        """Lazy-load Chatterbox TTS."""
        if self.use_turbo:
            if self._tts is None:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                self._tts = ChatterboxTurboTTS.from_pretrained(device=self.device)
            return self._tts
        else:
            if self._tts_full is None:
                from chatterbox.tts import ChatterboxTTS
                self._tts_full = ChatterboxTTS.from_pretrained(device=self.device)
            return self._tts_full

    def clean_text(self, text: str) -> str:
        """Clean Thinker output for TTS consumption.

        Removes thinking tokens, tool calls, SSML-like tags, and
        converts paralinguistic markers to TTS-friendly format.
        """
        # Remove tool calls
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)

        # Remove thinking tags
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # Remove SSML-like tags but keep content
        text = re.sub(r"<speak[^>]*>", "", text)
        text = re.sub(r"</speak>", "", text)

        # Keep paralinguistic markers that Chatterbox might handle
        # [laugh], [chuckle], [pause], [sigh] etc.
        # These are actually handled well by some TTS systems

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def map_style(self, emotion_label: str = None, prosody: dict = None,
                  conditioning_vector: torch.Tensor = None) -> dict:
        """Map emotion/prosody to Chatterbox generation parameters."""

        # If we have a trained StyleMapper and conditioning vector, use it
        if self.style_mapper is not None and conditioning_vector is not None:
            with torch.no_grad():
                if conditioning_vector.dim() == 1:
                    conditioning_vector = conditioning_vector.unsqueeze(0)
                return self.style_mapper(conditioning_vector.float())

        # Fall back to rule-based mapping
        params = {
            "exaggeration": 0.5,
            "cfg_weight": 0.5,
            "temperature": 0.8,
        }

        if emotion_label:
            params["exaggeration"] = EMOTION_EXAGGERATION.get(emotion_label, 0.5)
            params["temperature"] = EMOTION_TEMPERATURE.get(emotion_label, 0.8)

        if prosody:
            # Energy affects exaggeration
            energy = prosody.get("energy", 1.0)
            params["exaggeration"] *= min(energy, 1.5)
            params["exaggeration"] = min(params["exaggeration"], 1.0)

            # Speed affects nothing directly for Chatterbox (it controls its own pacing)
            # But very fast/slow speech might benefit from different temperatures
            speed = prosody.get("speed", 1.0)
            if speed > 1.5:
                params["temperature"] = min(params["temperature"] + 0.1, 1.0)

        return params

    def generate(
        self,
        text: str,
        emotion_label: str = None,
        prosody: dict = None,
        conditioning_vector: torch.Tensor = None,
        voice_path: str = None,
    ) -> torch.Tensor:
        """Generate speech from text with emotion-aware style.

        Args:
            text: Raw text from Thinker (will be cleaned)
            emotion_label: Detected emotion (e.g., "happy", "angry")
            prosody: Prosody dict from probe (speed, pitch, energy, emphasis)
            conditioning_vector: Raw 14-dim vector from probe (for StyleMapper)
            voice_path: Override voice reference (uses default if None)

        Returns:
            Audio waveform tensor
        """
        voice = voice_path or self.voice_path
        assert voice is not None, "No voice reference provided"

        # Clean text for TTS
        clean = self.clean_text(text)
        if not clean:
            clean = "..."

        # Map emotion to style parameters
        style = self.map_style(emotion_label, prosody, conditioning_vector)

        # Generate with appropriate Chatterbox version
        if self.use_turbo:
            # Turbo: limited style controls
            wav = self.tts.generate(
                text=clean,
                audio_prompt_path=voice,
                temperature=style["temperature"],
            )
        else:
            # Full: has emotion_adv (exaggeration) and CFG
            wav = self.tts.generate(
                text=clean,
                audio_prompt_path=voice,
                exaggeration=style["exaggeration"],
                cfg_weight=style["cfg_weight"],
                temperature=style["temperature"],
            )

        return wav

    @property
    def sr(self) -> int:
        """Sample rate of generated audio."""
        return self.tts.sr
