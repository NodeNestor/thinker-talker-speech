#!/usr/bin/env python3
"""Quick test: verify all pre-trained components load and run.

Run this before training to make sure everything works.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_thinker():
    print("=" * 50)
    print("Testing Thinker (Qwen 3.5)...")
    from src.model.thinker import Thinker

    thinker = Thinker(use_lora=False, device="cuda")
    print(f"  Hidden size: {thinker.hidden_size}")
    print(f"  Num layers: {thinker.num_layers}")

    # Test forward pass
    tokens = thinker.tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
    output = thinker(**tokens)

    # Check hidden states captured
    dn_states = thinker.get_deltanet_states()
    attn_states = thinker.get_attention_states()
    print(f"  DeltaNet layers captured: {len(dn_states)}")
    print(f"  Attention layers captured: {len(attn_states)}")
    if dn_states:
        first_key = list(dn_states.keys())[0]
        print(f"  DeltaNet state shape: {dn_states[first_key].shape}")
    if attn_states:
        first_key = list(attn_states.keys())[0]
        print(f"  Attention state shape: {attn_states[first_key].shape}")

    print("  PASS")
    del thinker
    torch.cuda.empty_cache()


def test_whisper_adapter():
    print("=" * 50)
    print("Testing Whisper Adapter...")
    from src.model.adapter import WhisperAdapter

    adapter = WhisperAdapter(thinker_hidden_size=1024, device="cuda")

    # Test with random audio (3 seconds at 16kHz)
    audio = torch.randn(1, 48000).to("cuda")
    embeds = adapter(audio=audio)
    print(f"  Input: 3s audio at 16kHz")
    print(f"  Output shape: {embeds.shape}")
    print(f"  Output rate: {adapter.output_rate_hz}Hz")
    print("  PASS")
    del adapter
    torch.cuda.empty_cache()


def test_emotion_probe():
    print("=" * 50)
    print("Testing Emotion Probe...")
    from src.model.emotion_probe import EmotionProbe

    probe = EmotionProbe(hidden_size=1024).to("cuda")

    # Fake hidden states
    dn_states = {i: torch.randn(1, 10, 1024).to("cuda") for i in range(18)}
    attn_states = {i: torch.randn(1, 10, 1024).to("cuda") for i in [3, 7, 11, 15, 19, 23]}

    result = probe(dn_states, attn_states)
    print(f"  Emotion: {result['emotion_label']}")
    print(f"  Prosody: {result['prosody_labels']}")
    print(f"  Conditioning dim: {probe.conditioning_dim}")
    print(f"  Conditioning shape: {result['conditioning_vector'].shape}")
    print("  PASS")


def test_speaker_encoder():
    print("=" * 50)
    print("Testing Speaker Encoder (ECAPA-TDNN)...")
    try:
        from src.model.speaker_encoder import SpeakerEncoder

        encoder = SpeakerEncoder(device="cuda")
        audio = torch.randn(48000)  # 3s at 16kHz
        embedding = encoder(audio, sample_rate=16000)
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding dim: {SpeakerEncoder.EMBEDDING_DIM}")
        print("  PASS")
    except ImportError:
        print("  SKIP (install speechbrain)")


def test_connector():
    print("=" * 50)
    print("Testing Thinker->Talker Connector...")
    from src.model.connector import ThinkerTalkerConnector

    connector = ThinkerTalkerConnector(
        thinker_hidden_size=1024,
        talker_hidden_size=512,
        emotion_dim=14,
        speaker_dim=192,
    ).to("cuda")

    thinker_hidden = torch.randn(1, 20, 1024).to("cuda")
    emotion_vec = torch.randn(1, 14).to("cuda")
    speaker_emb = torch.randn(1, 192).to("cuda")

    output = connector(thinker_hidden, emotion_vec, speaker_emb)
    print(f"  Input: thinker_hidden {thinker_hidden.shape}")
    print(f"  Output: {output.shape}")
    params = sum(p.numel() for p in connector.parameters())
    print(f"  Connector params: {params / 1e6:.1f}M")
    print("  PASS")


def main():
    print("Component Tests")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, some tests may fail")

    test_emotion_probe()
    test_connector()

    # These download models — run only if models/ exists
    if os.path.exists("models/thinker") or "--all" in sys.argv:
        test_thinker()
    else:
        print("\n[SKIP] Thinker test (run scripts/download_models.py first, or use --all)")

    if os.path.exists("models/whisper-small") or "--all" in sys.argv:
        test_whisper_adapter()
    else:
        print("[SKIP] Whisper test (run scripts/download_models.py first, or use --all)")

    if os.path.exists("models/ecapa-tdnn") or "--all" in sys.argv:
        test_speaker_encoder()
    else:
        print("[SKIP] Speaker encoder test (run scripts/download_models.py first)")

    print("\n" + "=" * 50)
    print("All tests passed!")


if __name__ == "__main__":
    main()
