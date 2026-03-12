"""Thinker->Talker Connector — bridges hidden states to the Talker's input space.

The connector takes:
  1. Thinker hidden states (text semantics)
  2. Emotion/prosody conditioning vector (from the probe)
  3. Speaker embedding (from ECAPA-TDNN)

And produces a unified conditioning signal for the Talker via
Adaptive Layer Normalization (AdaLN).

This is the key integration point — it's small (~5M params) and
is the main thing we train in Stage 4.
"""

import torch
import torch.nn as nn


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization — conditions normalization on a style vector.

    Used to inject speaker identity and emotion into the Talker.
    scale = Linear(style) + 1, shift = Linear(style)
    output = scale * LayerNorm(x) + shift
    """

    def __init__(self, hidden_size: int, conditioning_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.scale_proj = nn.Linear(conditioning_size, hidden_size)
        self.shift_proj = nn.Linear(conditioning_size, hidden_size)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, hidden]
            conditioning: [batch, conditioning_size]
        """
        normed = self.norm(x)
        # Expand conditioning to match sequence dim
        scale = self.scale_proj(conditioning).unsqueeze(1) + 1.0
        shift = self.shift_proj(conditioning).unsqueeze(1)
        return normed * scale + shift


class ThinkerTalkerConnector(nn.Module):
    """Connects Thinker output to Talker input.

    Merges three signals into a single conditioning tensor:
    - Thinker hidden states (semantic content)
    - Emotion/prosody vector (how to say it)
    - Speaker embedding (whose voice to use)
    """

    def __init__(
        self,
        thinker_hidden_size: int = 1024,
        talker_hidden_size: int = 512,  # Chatterbox Turbo or custom Talker
        emotion_dim: int = 14,          # 10 emotions + 4 prosody
        speaker_dim: int = 192,         # ECAPA-TDNN
    ):
        super().__init__()

        # Project Thinker hidden states to Talker dimension
        self.content_proj = nn.Sequential(
            nn.Linear(thinker_hidden_size, talker_hidden_size),
            nn.SiLU(),
            nn.Linear(talker_hidden_size, talker_hidden_size),
        )

        # Fuse emotion + speaker into a style vector
        style_dim = emotion_dim + speaker_dim
        self.style_fuse = nn.Sequential(
            nn.Linear(style_dim, talker_hidden_size),
            nn.SiLU(),
            nn.Linear(talker_hidden_size, talker_hidden_size),
        )

        # AdaLN: condition content with style
        self.adaln = AdaptiveLayerNorm(talker_hidden_size, talker_hidden_size)

    def forward(
        self,
        thinker_hidden: torch.Tensor,       # [batch, seq, thinker_hidden]
        emotion_vector: torch.Tensor,        # [batch, emotion_dim]
        speaker_embedding: torch.Tensor,     # [batch, speaker_dim]
    ) -> torch.Tensor:
        """
        Returns:
            talker_input: [batch, seq, talker_hidden] — ready for Talker
        """
        # Project content
        content = self.content_proj(thinker_hidden)  # [batch, seq, talker_hidden]

        # Fuse style signals
        style = torch.cat([emotion_vector, speaker_embedding], dim=-1)
        style = self.style_fuse(style)  # [batch, talker_hidden]

        # Apply AdaLN — condition content with style
        output = self.adaln(content, style)

        return output
