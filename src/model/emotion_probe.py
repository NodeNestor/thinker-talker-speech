"""Emotion & Prosody Probe — reads emotion/tone/pace from Thinker hidden states.

Two probe types that work together:

1. DeltaNetProbe: Reads the DeltaNet (linear attention) recurrent states.
   These accumulate context over time, making them ideal for extracting
   slow-moving features like overall mood, conversational energy, and
   speaking pace. Think of it as "what's the vibe of this conversation?"

2. AttentionProbe: Reads the standard attention layer hidden states.
   These capture what the model is focusing on RIGHT NOW — good for
   per-token features like word emphasis, surprise, and local emotion shifts.

Combined output: a continuous emotion/prosody vector that controls the Talker.

The probes are tiny (~1-5M params) and run in parallel with the Thinker,
adding zero latency to the main inference path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Emotion categories (can be extended)
EMOTION_LABELS = [
    "neutral", "happy", "sad", "angry", "excited",
    "fearful", "surprised", "disgusted", "calm", "confused",
]

# Prosody dimensions (continuous floats)
PROSODY_DIMS = [
    "speed",       # speaking rate multiplier (0.5 = slow, 1.0 = normal, 2.0 = fast)
    "pitch",       # pitch shift (-1 = low, 0 = normal, 1 = high)
    "energy",      # loudness/intensity (0 = whisper, 1 = normal, 2 = shouting)
    "emphasis",    # word stress level (0 = flat, 1 = emphasized)
]


class DeltaNetProbe(nn.Module):
    """Probe that reads DeltaNet recurrent states for mood/energy/pace.

    DeltaNet layers maintain a state matrix that's updated at each token.
    This state is a compressed summary of all prior context — perfect for
    extracting "slow" features like overall mood and conversational energy.

    We pool across multiple DeltaNet layers (early = syntax, mid = semantics,
    late = pragmatics/emotion) and learn a mapping to emotion+prosody space.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_emotions: int = len(EMOTION_LABELS),
        num_prosody: int = len(PROSODY_DIMS),
        probe_hidden: int = 256,
        num_layers_to_probe: int = 6,  # How many DeltaNet layers to read
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.num_prosody = num_prosody

        # Layer attention: learn which DeltaNet layers matter most
        self.layer_weights = nn.Parameter(torch.ones(num_layers_to_probe))

        # Pool across sequence dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Emotion head: hidden_size -> emotion logits
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, probe_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(probe_hidden, probe_hidden),
            nn.SiLU(),
            nn.Linear(probe_hidden, num_emotions),
        )

        # Prosody head: hidden_size -> continuous prosody values
        self.prosody_head = nn.Sequential(
            nn.Linear(hidden_size, probe_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(probe_hidden, probe_hidden),
            nn.SiLU(),
            nn.Linear(probe_hidden, num_prosody),
        )

    def forward(self, deltanet_states: dict[int, torch.Tensor]) -> dict:
        """
        Args:
            deltanet_states: {layer_idx: tensor [batch, seq, hidden]} from Thinker

        Returns:
            emotion_logits: [batch, num_emotions]
            emotion_probs: [batch, num_emotions] (softmax)
            prosody: [batch, num_prosody] (continuous values)
        """
        if not deltanet_states:
            raise ValueError("No DeltaNet states provided")

        # Stack available layers
        layers = sorted(deltanet_states.keys())
        # Take last N layers (late layers = more semantic/pragmatic)
        layers = layers[-self.layer_weights.shape[0]:]

        stacked = torch.stack([deltanet_states[l] for l in layers], dim=0)
        # stacked: [num_layers, batch, seq, hidden]

        # Weighted sum across layers
        weights = F.softmax(self.layer_weights[:len(layers)], dim=0)
        # weights: [num_layers]
        pooled = torch.einsum("l,lbsh->bsh", weights, stacked)
        # pooled: [batch, seq, hidden]

        # Pool across sequence (take mean — represents overall context)
        pooled = pooled.mean(dim=1)  # [batch, hidden]

        # Emotion prediction
        emotion_logits = self.emotion_head(pooled)
        emotion_probs = F.softmax(emotion_logits, dim=-1)

        # Prosody prediction (sigmoid for bounded output, then scale)
        prosody_raw = self.prosody_head(pooled)
        prosody = torch.sigmoid(prosody_raw)
        # Scale: speed [0.5, 2.0], pitch [-1, 1], energy [0, 2], emphasis [0, 1]
        prosody_scaled = prosody.clone()
        prosody_scaled[:, 0] = prosody[:, 0] * 1.5 + 0.5   # speed: 0.5-2.0
        prosody_scaled[:, 1] = prosody[:, 1] * 2.0 - 1.0    # pitch: -1 to 1
        prosody_scaled[:, 2] = prosody[:, 2] * 2.0           # energy: 0-2
        # emphasis stays 0-1

        return {
            "emotion_logits": emotion_logits,
            "emotion_probs": emotion_probs,
            "emotion_label": EMOTION_LABELS[emotion_probs.argmax(dim=-1).item()]
                             if emotion_probs.shape[0] == 1 else None,
            "prosody": prosody_scaled,
            "prosody_labels": dict(zip(PROSODY_DIMS, prosody_scaled[0].tolist()))
                              if prosody_scaled.shape[0] == 1 else None,
        }


class AttentionProbe(nn.Module):
    """Probe that reads standard attention layers for per-token emphasis.

    Standard attention layers (every 4th in Qwen 3.5) show what the model
    is attending to at each position. This gives us per-token features:
    - Which words to emphasize
    - Local surprise / unexpectedness
    - Immediate emotional shifts within a sentence
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_prosody: int = len(PROSODY_DIMS),
        probe_hidden: int = 128,
    ):
        super().__init__()

        # Per-token prosody adjustment (additive delta on top of DeltaNet probe)
        self.token_prosody = nn.Sequential(
            nn.Linear(hidden_size, probe_hidden),
            nn.SiLU(),
            nn.Linear(probe_hidden, num_prosody),
            nn.Tanh(),  # output in [-1, 1] as adjustment
        )

    def forward(self, attention_states: dict[int, torch.Tensor]) -> dict:
        """
        Args:
            attention_states: {layer_idx: tensor [batch, seq, hidden]}

        Returns:
            token_prosody_delta: [batch, seq, num_prosody] — per-token adjustments
        """
        if not attention_states:
            return {"token_prosody_delta": None}

        # Use the last attention layer
        last_layer = max(attention_states.keys())
        hs = attention_states[last_layer]  # [batch, seq, hidden]

        delta = self.token_prosody(hs)  # [batch, seq, num_prosody]
        # Scale deltas to be small adjustments
        delta = delta * 0.3

        return {"token_prosody_delta": delta}


class EmotionProbe(nn.Module):
    """Combined emotion/prosody probe using both DeltaNet and attention states.

    This is the main interface. It combines:
    - DeltaNet states -> overall mood + base prosody (slow context)
    - Attention states -> per-token prosody adjustments (fast context)

    Output: emotion vector + prosody vector ready to inject into the Talker.
    """

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.deltanet_probe = DeltaNetProbe(hidden_size=hidden_size)
        self.attention_probe = AttentionProbe(hidden_size=hidden_size)

    def forward(
        self,
        deltanet_states: dict[int, torch.Tensor],
        attention_states: dict[int, torch.Tensor],
    ) -> dict:
        """
        Returns:
            emotion_probs: [batch, num_emotions] — overall emotional state
            base_prosody: [batch, num_prosody] — base speaking style
            token_prosody_delta: [batch, seq, num_prosody] — per-token adjustments
            conditioning_vector: [batch, num_emotions + num_prosody] — for Talker AdaLN
        """
        # Overall mood from DeltaNet
        delta_out = self.deltanet_probe(deltanet_states)

        # Per-token adjustments from attention
        attn_out = self.attention_probe(attention_states)

        # Combine into a single conditioning vector for the Talker
        conditioning = torch.cat([
            delta_out["emotion_probs"],
            delta_out["prosody"],
        ], dim=-1)  # [batch, num_emotions + num_prosody]

        return {
            **delta_out,
            **attn_out,
            "conditioning_vector": conditioning,
        }

    @property
    def conditioning_dim(self) -> int:
        return len(EMOTION_LABELS) + len(PROSODY_DIMS)
