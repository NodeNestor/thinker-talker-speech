"""Speaker Encoder — extract voice embeddings from short audio samples.

Uses ECAPA-TDNN (SpeechBrain) to compute a 192-dim speaker embedding
from as little as 3 seconds of reference audio. The embedding captures
timbre, pitch range, and speaking style.

Voice embeddings are injected into the Talker via Adaptive Layer Norm (AdaLN)
and can be swapped at runtime — no retraining needed.
"""

import torch
import torch.nn as nn
import torchaudio


class SpeakerEncoder(nn.Module):
    """ECAPA-TDNN speaker encoder from SpeechBrain."""

    EMBEDDING_DIM = 192

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

        # Load pre-trained ECAPA-TDNN
        from speechbrain.inference.speaker import EncoderClassifier
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/ecapa-tdnn",
            run_opts={"device": device},
        )

        # Freeze — we never train this
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Args:
            audio: Waveform [batch, samples] or [samples] at any sample rate
            sample_rate: Input sample rate (will resample to 16kHz if needed)

        Returns:
            embedding: [batch, 192] speaker embedding
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(self.device)
            audio = resampler(audio)

        embedding = self.encoder.encode_batch(audio.to(self.device))
        return embedding.squeeze(1)  # [batch, 192]

    @torch.no_grad()
    def from_file(self, audio_path: str) -> torch.Tensor:
        """Load audio file and compute speaker embedding."""
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)  # mono
        return self.forward(audio, sample_rate=sr)
