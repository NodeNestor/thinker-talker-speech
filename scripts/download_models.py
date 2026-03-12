#!/usr/bin/env python3
"""Download all pre-trained model components."""

import os
import sys

def main():
    os.makedirs("models", exist_ok=True)

    print("=" * 60)
    print("Downloading pre-trained components")
    print("=" * 60)

    # 1. Qwen 3.5 0.6B (Thinker)
    print("\n[1/4] Downloading Qwen 3.5 Thinker...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "Qwen/Qwen3.5-0.6B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.save_pretrained("models/thinker")
        model.save_pretrained("models/thinker")
        print(f"  Saved to models/thinker ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")
    except Exception as e:
        print(f"  Qwen3.5-0.6B not available, trying fallback Qwen3-0.6B...")
        model_id = "Qwen/Qwen3-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.save_pretrained("models/thinker")
        model.save_pretrained("models/thinker")
        print(f"  Saved to models/thinker (fallback: {model_id})")

    del model, tokenizer

    # 2. Whisper-small (Speech encoder)
    print("\n[2/4] Downloading Whisper-small...")
    from transformers import WhisperModel, WhisperProcessor
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    whisper = WhisperModel.from_pretrained("openai/whisper-small")
    processor.save_pretrained("models/whisper-small")
    whisper.save_pretrained("models/whisper-small")
    print(f"  Saved to models/whisper-small ({sum(p.numel() for p in whisper.parameters()) / 1e6:.0f}M params)")
    del whisper, processor

    # 3. ECAPA-TDNN speaker encoder
    print("\n[3/4] Downloading ECAPA-TDNN speaker encoder...")
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/ecapa-tdnn",
        )
        print("  Saved to models/ecapa-tdnn (192-dim embeddings)")
        del encoder
    except Exception as e:
        print(f"  Warning: SpeechBrain not installed or download failed: {e}")
        print("  Install with: pip install speechbrain")

    # 4. Chatterbox Turbo (Talker) — or just download the config for now
    print("\n[4/4] Downloading Chatterbox Turbo (Talker)...")
    try:
        from transformers import AutoModel
        # Try to download — may need custom code
        print("  Checking ResembleAI/chatterbox-turbo...")
        # Chatterbox uses its own loading mechanism
        # pip install chatterbox-tts
        print("  Install Chatterbox: pip install chatterbox-tts")
        print("  It will download on first use (~350M)")
    except Exception as e:
        print(f"  Note: Install chatterbox-tts separately: pip install chatterbox-tts")

    print("\n" + "=" * 60)
    print("Done! All models saved to models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
