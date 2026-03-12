"""Whisper Adapter — maps Whisper encoder features to Thinker embedding space.

Architecture: CNN downsampler (50Hz -> 25Hz) + LayerNorm + Linear projection.
The adapter is the main trainable component in Stage 1.
"""

import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperProcessor


WHISPER_MODEL_ID = "openai/whisper-small"


class WhisperAdapter(nn.Module):
    """Whisper encoder + CNN adapter to Thinker embedding space."""

    def __init__(
        self,
        thinker_hidden_size: int = 1024,
        whisper_model_id: str = WHISPER_MODEL_ID,
        freeze_encoder: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # Load Whisper encoder (frozen)
        self.processor = WhisperProcessor.from_pretrained(whisper_model_id)
        whisper = WhisperModel.from_pretrained(
            whisper_model_id, torch_dtype=dtype
        ).to(device)
        self.encoder = whisper.encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        whisper_dim = self.encoder.config.d_model  # 768 for whisper-small

        # CNN downsampler: 50Hz -> 25Hz (stride 2)
        self.downsample = nn.Sequential(
            nn.Conv1d(whisper_dim, thinker_hidden_size, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )

        # Projection to Thinker hidden size
        self.proj = nn.Sequential(
            nn.LayerNorm(thinker_hidden_size),
            nn.Linear(thinker_hidden_size, thinker_hidden_size),
            nn.SiLU(),
            nn.Linear(thinker_hidden_size, thinker_hidden_size),
        )

    def preprocess(self, audio: torch.Tensor, sample_rate: int = 16000) -> dict:
        """Preprocess raw audio waveform for Whisper."""
        inputs = self.processor(
            audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        return {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}

    def forward(self, audio: torch.Tensor = None, whisper_features: torch.Tensor = None):
        """
        Args:
            audio: Raw waveform [batch, samples] at 16kHz
            whisper_features: Pre-computed Whisper input features [batch, mel_bins, time]

        Returns:
            embeddings: [batch, time//2, thinker_hidden_size] at ~25Hz
        """
        if whisper_features is None:
            if audio is None:
                raise ValueError("Provide either audio or whisper_features")
            inputs = self.preprocess(audio)
            whisper_features = inputs["input_features"]

        # Whisper encoder: [batch, time, 768] at 50Hz
        encoder_out = self.encoder(whisper_features).last_hidden_state

        # CNN downsample: transpose for Conv1d [batch, 768, time] -> [batch, hidden, time//2]
        x = self.downsample(encoder_out.transpose(1, 2))

        # Back to [batch, time//2, hidden]
        x = x.transpose(1, 2)

        # Project
        x = self.proj(x)

        return x

    @property
    def output_rate_hz(self) -> int:
        """Output frame rate in Hz."""
        return 25
