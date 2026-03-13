"""Thinker->Talker Connector — projects Thinker hidden states into T3's embedding space.

Two modes of operation:

1. **Direct mode (trained connector)**: Thinker hidden states are projected directly
   into T3's text embedding space, bypassing text decoding entirely.
   This preserves emotion/prosody information that gets lost in text.

   Thinker hidden states [B, seq, 1024] → HiddenStateConnector → T3 text embeddings [B, seq', 1024]
   These replace t3.text_emb(text_tokens) in T3's prepare_input_embeds.

2. **Text mode (rule-based fallback)**: Thinker generates text, which is re-tokenized
   by T3's tokenizer and passed through normally. Style params (temperature, etc.)
   are set based on emotion label lookup tables.

Both Thinker (Qwen 3.5) and T3 Turbo (GPT2-medium) have 1024-dim hidden states,
making the projection natural. The connector learns to align the two spaces using
paired (text, hidden_state) data — same approach as the Whisper adapter (Stage 1).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from .emotion_probe import EMOTION_LABELS, PROSODY_DIMS


# ── Rule-based style mapping (fallback) ──────────────────────────────

EMOTION_EXAGGERATION = {
    "neutral": 0.3, "happy": 0.6, "sad": 0.5, "angry": 0.8, "excited": 0.9,
    "fearful": 0.7, "surprised": 0.8, "disgusted": 0.6, "calm": 0.2, "confused": 0.5,
}

EMOTION_TEMPERATURE = {
    "neutral": 0.7, "happy": 0.85, "sad": 0.75, "angry": 0.9, "excited": 0.95,
    "fearful": 0.8, "surprised": 0.9, "disgusted": 0.75, "calm": 0.65, "confused": 0.8,
}


# ── Neural connector: Thinker hidden states → T3 text embeddings ─────

class HiddenStateConnector(nn.Module):
    """Projects Thinker hidden states into T3's text embedding space.

    Replaces the text path entirely: instead of Thinker → text → T3 tokenize → T3 embed,
    this goes Thinker → connector → T3 directly.

    Both models have 1024-dim hidden states, so the projection is natural.
    A small MLP with residual connections learns the alignment.

    Training:
        For each text, compute both:
          - Thinker hidden states (from forward pass)
          - T3 text embeddings (from T3's own tokenizer + embedding layer)
        Loss: MSE + cosine similarity between projected states and T3 embeddings.
        Sequence alignment handled by learned cross-attention (different tokenizers).

    ~2M params. Trained in Stage 4.
    """

    def __init__(self, thinker_dim: int = 1024, t3_dim: int = 1024,
                 emotion_dim: int = 14, hidden_dim: int = 512):
        super().__init__()
        self.thinker_dim = thinker_dim
        self.t3_dim = t3_dim

        # Emotion conditioning via FiLM (feature-wise linear modulation)
        # Injects emotion/prosody info into the projection
        self.emotion_scale = nn.Linear(emotion_dim, thinker_dim)
        self.emotion_shift = nn.Linear(emotion_dim, thinker_dim)

        # Main projection: Thinker space → T3 space
        self.proj = nn.Sequential(
            nn.LayerNorm(thinker_dim),
            nn.Linear(thinker_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, t3_dim),
        )

        # Residual projection (if dims match, add a learned residual)
        self.use_residual = (thinker_dim == t3_dim)
        if self.use_residual:
            self.residual_gate = nn.Parameter(torch.zeros(1))

        # Sequence alignment: cross-attention to handle tokenizer differences
        # Thinker and T3 use different tokenizers → different sequence lengths
        self.align = nn.MultiheadAttention(
            embed_dim=t3_dim, num_heads=8, batch_first=True,
        )
        self.align_norm = nn.LayerNorm(t3_dim)

    def forward(self, thinker_hidden: torch.Tensor,
                emotion_vector: torch.Tensor = None,
                target_length: int = None) -> torch.Tensor:
        """Project Thinker hidden states to T3 text embedding space.

        Args:
            thinker_hidden: [batch, seq_thinker, 1024] from Thinker's last hidden state
            emotion_vector: [batch, 14] from EmotionProbe (optional)
            target_length: desired output sequence length (T3's text token count).
                          If None, output has same length as input.

        Returns:
            t3_text_emb: [batch, seq_t3, 1024] — drop-in replacement for t3.text_emb(tokens)
        """
        x = thinker_hidden.float()

        # FiLM conditioning: modulate features with emotion
        if emotion_vector is not None:
            ev = emotion_vector.float()
            scale = self.emotion_scale(ev).unsqueeze(1)  # [B, 1, dim]
            shift = self.emotion_shift(ev).unsqueeze(1)   # [B, 1, dim]
            x = x * (1 + scale) + shift

        # Project to T3 space
        projected = self.proj(x)  # [B, seq_thinker, t3_dim]

        # Residual connection (Thinker and T3 share 1024-dim)
        if self.use_residual:
            gate = torch.sigmoid(self.residual_gate)
            projected = projected + gate * thinker_hidden.float()

        # Align to target sequence length if needed
        if target_length is not None and target_length != projected.shape[1]:
            # Interpolate to target length, then refine with self-attention
            projected = F.interpolate(
                projected.transpose(1, 2),  # [B, dim, seq]
                size=target_length,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)  # [B, target_len, dim]

            # Self-attention to refine interpolated positions
            refined, _ = self.align(projected, projected, projected)
            projected = self.align_norm(projected + refined)

        return projected

    def compute_loss(self, projected: torch.Tensor,
                     target_t3_emb: torch.Tensor) -> dict:
        """Compute training loss between projected and target T3 embeddings.

        Args:
            projected: [batch, seq, dim] — connector output
            target_t3_emb: [batch, seq, dim] — T3's actual text embeddings

        Returns:
            dict with total_loss, mse_loss, cosine_loss
        """
        # MSE loss (per-element)
        mse = F.mse_loss(projected, target_t3_emb)

        # Cosine similarity loss (per-token)
        cos_sim = F.cosine_similarity(projected, target_t3_emb, dim=-1)
        cosine_loss = (1 - cos_sim).mean()

        total = mse + 0.5 * cosine_loss

        return {
            "total_loss": total,
            "mse_loss": mse,
            "cosine_loss": cosine_loss,
        }


class StyleMapper(nn.Module):
    """Learned mapping from emotion/prosody vector to Chatterbox style parameters.

    Maps the probe's 14-dim conditioning vector (10 emotions + 4 prosody) to:
      - exaggeration: float [0, 1] — emotion intensity for Chatterbox
      - cfg_weight: float [0, 1] — classifier-free guidance strength
      - temperature: float [0.5, 1.2] — sampling diversity

    Small network (~10K params).
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
        raw = self.net(conditioning_vector)
        exaggeration = torch.sigmoid(raw[:, 0])
        cfg_weight = torch.sigmoid(raw[:, 1]) * 0.8
        temperature = torch.sigmoid(raw[:, 2]) * 0.7 + 0.5

        return {
            "exaggeration": exaggeration.mean().item(),
            "cfg_weight": cfg_weight.mean().item(),
            "temperature": temperature.mean().item(),
        }


# ── Pipeline connector (wraps both modes) ────────────────────────────

class ThinkerTalkerConnector:
    """Connects Thinker output to Chatterbox TTS.

    Two modes:
    1. Direct mode: HiddenStateConnector projects hidden states → T3 embeddings
       (skips text decoding, preserves emotion/prosody info)
    2. Text mode: standard text → T3 tokenizer → T3 embeddings
       (rule-based style mapping from emotion labels)

    Usage:
        # Text mode (fallback)
        connector = ThinkerTalkerConnector(voice_path="speaker.wav")
        audio = connector.generate(text="Hello!", emotion_label="happy")

        # Direct mode (trained connector)
        connector = ThinkerTalkerConnector(
            voice_path="speaker.wav",
            hidden_connector=trained_connector,
        )
        audio = connector.generate_from_hidden(
            thinker_hidden=hidden_states,
            emotion_vector=conditioning_vector,
        )
    """

    def __init__(
        self,
        voice_path: str = None,
        device: str = "cuda",
        use_turbo: bool = True,
        style_mapper: StyleMapper = None,
        hidden_connector: HiddenStateConnector = None,
    ):
        self.voice_path = voice_path
        self.device = device
        self.use_turbo = use_turbo
        self.style_mapper = style_mapper
        self.hidden_connector = hidden_connector
        self._tts = None
        self._tts_full = None

    @property
    def has_direct_mode(self) -> bool:
        """Whether the trained hidden-state connector is available."""
        return self.hidden_connector is not None

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
        """Clean Thinker output for TTS consumption."""
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<speak[^>]*>", "", text)
        text = re.sub(r"</speak>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def map_style(self, emotion_label: str = None, prosody: dict = None,
                  conditioning_vector: torch.Tensor = None) -> dict:
        """Map emotion/prosody to Chatterbox generation parameters."""
        if self.style_mapper is not None and conditioning_vector is not None:
            with torch.no_grad():
                if conditioning_vector.dim() == 1:
                    conditioning_vector = conditioning_vector.unsqueeze(0)
                return self.style_mapper(conditioning_vector.float())

        params = {
            "exaggeration": 0.5,
            "cfg_weight": 0.5,
            "temperature": 0.8,
        }

        if emotion_label:
            params["exaggeration"] = EMOTION_EXAGGERATION.get(emotion_label, 0.5)
            params["temperature"] = EMOTION_TEMPERATURE.get(emotion_label, 0.8)

        if prosody:
            energy = prosody.get("energy", 1.0)
            params["exaggeration"] *= min(energy, 1.5)
            params["exaggeration"] = min(params["exaggeration"], 1.0)
            speed = prosody.get("speed", 1.0)
            if speed > 1.5:
                params["temperature"] = min(params["temperature"] + 0.1, 1.0)

        return params

    def generate(self, text: str, emotion_label: str = None, prosody: dict = None,
                 conditioning_vector: torch.Tensor = None,
                 voice_path: str = None) -> torch.Tensor:
        """Generate speech from text (text mode)."""
        voice = voice_path or self.voice_path
        assert voice is not None, "No voice reference provided"

        clean = self.clean_text(text)
        if not clean:
            clean = "..."

        style = self.map_style(emotion_label, prosody, conditioning_vector)

        if self.use_turbo:
            wav = self.tts.generate(text=clean, audio_prompt_path=voice,
                                   temperature=style["temperature"])
        else:
            wav = self.tts.generate(text=clean, audio_prompt_path=voice,
                                   exaggeration=style["exaggeration"],
                                   cfg_weight=style["cfg_weight"],
                                   temperature=style["temperature"])
        return wav

    @torch.inference_mode()
    def generate_from_hidden(
        self,
        thinker_hidden: torch.Tensor,
        emotion_vector: torch.Tensor = None,
        voice_path: str = None,
        temperature: float = 0.8,
    ) -> torch.Tensor:
        """Generate speech directly from Thinker hidden states (direct mode).

        Skips text decoding — hidden states are projected into T3's embedding space
        and fed directly to the speech generation pipeline.

        Args:
            thinker_hidden: [1, seq, 1024] — Thinker's last hidden state
            emotion_vector: [1, 14] — from EmotionProbe
            voice_path: reference speaker audio
            temperature: sampling temperature

        Returns:
            Audio waveform tensor
        """
        assert self.hidden_connector is not None, "No trained HiddenStateConnector"
        voice = voice_path or self.voice_path
        assert voice is not None, "No voice reference provided"

        # Project hidden states to T3 text embedding space
        t3_text_emb = self.hidden_connector(thinker_hidden, emotion_vector)
        # t3_text_emb: [1, seq, 1024] — replaces t3.text_emb(text_tokens)

        # Get T3 and voice conditioning
        tts = self.tts
        tts.prepare_conditionals(voice, exaggeration=0.0)
        t3 = tts.t3
        conds = tts.conds

        # Build T3 input: [conditioning, projected_text, speech_start]
        cond_emb = t3.prepare_conditioning(conds.t3)
        speech_start = t3.speech_emb(
            torch.tensor([[t3.hp.start_speech_token]], device=self.device)
        )

        embeds = torch.cat([cond_emb, t3_text_emb, speech_start], dim=1)

        # Run T3 generation (same as StreamingTTSPipeline but non-streaming)
        from transformers.generation.logits_process import (
            LogitsProcessorList, TemperatureLogitsWarper,
            TopKLogitsWarper, RepetitionPenaltyLogitsProcessor,
        )

        logits_processors = LogitsProcessorList([
            TemperatureLogitsWarper(temperature),
            TopKLogitsWarper(1000),
            RepetitionPenaltyLogitsProcessor(1.2),
        ])

        # Initial forward
        outputs = t3.tfmr(inputs_embeds=embeds, use_cache=True)
        past_kv = outputs.past_key_values
        speech_logits = t3.speech_head(outputs[0][:, -1:])

        start_token = torch.tensor([[t3.hp.start_speech_token]], device=self.device)
        processed = logits_processors(start_token, speech_logits[:, -1, :])
        probs = F.softmax(processed, dim=-1)
        current = torch.multinomial(probs, num_samples=1)

        all_tokens = [current]

        for _ in range(1000):
            if torch.all(current == t3.hp.stop_speech_token):
                break

            current_emb = t3.speech_emb(current)
            outputs = t3.tfmr(inputs_embeds=current_emb,
                             past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            speech_logits = t3.speech_head(outputs[0])

            input_ids = torch.cat(all_tokens, dim=1)
            processed = logits_processors(input_ids, speech_logits[:, -1, :])
            if torch.all(processed == -float("inf")):
                break

            probs = F.softmax(processed, dim=-1)
            current = torch.multinomial(probs, num_samples=1)
            all_tokens.append(current)

        # Vocode
        from chatterbox.models.s3gen.const import S3GEN_SIL
        silence = [torch.tensor([[S3GEN_SIL]], device=self.device)] * 3
        tokens = torch.cat(all_tokens + silence, dim=1).clamp(max=6560)

        import numpy as np
        wav, _ = tts.s3gen.inference(
            speech_tokens=tokens, ref_dict=conds.gen, n_cfm_timesteps=2,
        )
        wav = wav.squeeze(0).cpu().numpy().astype(np.float32)
        wav = tts.watermarker.apply_watermark(wav, sample_rate=tts.sr)

        return torch.from_numpy(wav)

    @property
    def sr(self) -> int:
        """Sample rate of generated audio."""
        return self.tts.sr
