#!/usr/bin/env python3
"""Download all pre-trained model components.

Components:
  1. Whisper-small (244M) — speech encoder (frozen)
  2. ECAPA-TDNN (~7M)    — speaker encoder for voice cloning
  3. Chatterbox Turbo     — TTS talker (downloads on first use)

The Qwen 3.5 0.8B Thinker is loaded via Unsloth from a LoRA checkpoint,
not downloaded here. See README for LoRA training (Stage 2).
"""

import os


def main():
    print("=" * 60)
    print("Downloading pre-trained components")
    print("=" * 60)

    # 1. Whisper-small (Speech encoder)
    print("\n[1/3] Downloading Whisper-small...")
    from transformers import WhisperModel, WhisperProcessor
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    whisper = WhisperModel.from_pretrained("openai/whisper-small")
    print(f"  Cached ({sum(p.numel() for p in whisper.parameters()) / 1e6:.0f}M params)")
    del whisper, processor

    # 2. ECAPA-TDNN speaker encoder
    print("\n[2/3] Downloading ECAPA-TDNN speaker encoder...")
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/ecapa-tdnn",
        )
        print("  Cached (192-dim speaker embeddings)")
        del encoder
    except ImportError:
        print("  SpeechBrain not installed. Run: pip install speechbrain")

    # 3. Chatterbox Turbo (Talker)
    print("\n[3/3] Chatterbox Turbo TTS...")
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        print("  chatterbox-tts installed. Model downloads on first use (~350M).")
        print("  To pre-download: python -c \"from chatterbox.tts_turbo import ChatterboxTurboTTS; ChatterboxTurboTTS.from_pretrained('cpu')\"")
    except ImportError:
        print("  Not installed. Run: pip install chatterbox-tts")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
