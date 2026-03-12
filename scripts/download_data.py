#!/usr/bin/env python3
"""Download training datasets.

Datasets by training stage:
  Stage 1 (Adapter):  LibriSpeech clean-360
  Stage 2 (LoRA):     VoiceAssistant-400K
  Stage 3 (Probe):    GoEmotions + UltraVoice + Expresso
  Stage 4 (Connect):  LibriTTS-R clean
"""

import os
import sys
import argparse


def download_stage1():
    """LibriSpeech clean-360 for adapter pretraining."""
    from datasets import load_dataset
    print("[Stage 1] Downloading LibriSpeech clean-360...")
    ds = load_dataset("openslr/librispeech_asr", split="train.clean.360")
    ds.save_to_disk("data/librispeech_clean360")
    print(f"  Saved {len(ds)} samples to data/librispeech_clean360")


def download_stage2():
    """VoiceAssistant-400K for Thinker LoRA tuning."""
    from datasets import load_dataset
    print("[Stage 2] Downloading VoiceAssistant-400K...")
    ds = load_dataset("gpt-omni/VoiceAssistant-400K")
    ds.save_to_disk("data/voiceassistant_400k")
    print(f"  Saved to data/voiceassistant_400k")


def download_stage3():
    """Emotion/prosody datasets for the probe."""
    from datasets import load_dataset

    print("[Stage 3a] Downloading GoEmotions (28 text emotions)...")
    ds = load_dataset("google-research-datasets/go_emotions")
    ds.save_to_disk("data/goemotions")
    print(f"  Saved to data/goemotions")

    print("[Stage 3b] Downloading UltraVoice (833h, emotion+speed+volume)...")
    try:
        ds = load_dataset("tutu0604/UltraVoice")
        ds.save_to_disk("data/ultravoice")
        print(f"  Saved to data/ultravoice")
    except Exception as e:
        print(f"  Warning: {e}")
        print("  UltraVoice may require authentication or manual download")

    print("[Stage 3c] Downloading Expresso (styled speech)...")
    try:
        ds = load_dataset("ylacombe/expresso")
        ds.save_to_disk("data/expresso")
        print(f"  Saved to data/expresso")
    except Exception as e:
        print(f"  Warning: {e}")

    print("[Stage 3d] Downloading MELD (dialogue emotions)...")
    try:
        ds = load_dataset("ajyy/MELD_audio")
        ds.save_to_disk("data/meld")
        print(f"  Saved to data/meld")
    except Exception as e:
        print(f"  Warning: {e}")


def download_stage4():
    """LibriTTS-R for connector training."""
    from datasets import load_dataset
    print("[Stage 4] Downloading LibriTTS-R clean...")
    ds = load_dataset("mythicinfinity/libritts_r", "clean")
    ds.save_to_disk("data/libritts_r_clean")
    print(f"  Saved to data/libritts_r_clean")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Download data for specific stage only")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    if args.stage is None or args.stage == 1:
        download_stage1()
    if args.stage is None or args.stage == 2:
        download_stage2()
    if args.stage is None or args.stage == 3:
        download_stage3()
    if args.stage is None or args.stage == 4:
        download_stage4()

    print("\nDone!")


if __name__ == "__main__":
    main()
