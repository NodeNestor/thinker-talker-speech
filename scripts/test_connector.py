#!/usr/bin/env python3
"""Test end-to-end: Thinker generates text + emotion → Connector → Chatterbox speaks.

Runs several prompts through the full pipeline and saves audio output.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/test_connector.py
    CUDA_VISIBLE_DEVICES=1 python scripts/test_connector.py --text "Tell me something exciting!"
    CUDA_VISIBLE_DEVICES=1 python scripts/test_connector.py --no-turbo  # use full Chatterbox with emotion
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch torchaudio for SpeechBrain compatibility on newer torchaudio
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

import torch
import numpy as np
import soundfile as sf


def load_thinker_and_probe(lora_path, probe_ckpt, device="cuda"):
    """Load Thinker + Probe for emotion detection."""
    import unsloth
    from unsloth import FastLanguageModel
    from src.model.emotion_probe import EmotionProbe
    from src.training.train_probe import HiddenStateCapture

    print("Loading Thinker...")
    thinker, processor = FastLanguageModel.from_pretrained(
        lora_path, device_map=device, dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(thinker)
    thinker.eval()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    hidden_size = thinker.config.get_text_config().hidden_size

    capture = HiddenStateCapture(thinker)

    print("Loading Probe...")
    probe = EmotionProbe(hidden_size=hidden_size).to(device)
    if os.path.exists(probe_ckpt):
        probe.load_state_dict(torch.load(probe_ckpt, map_location=device))
    probe.eval()

    return thinker, tokenizer, capture, probe


def run_thinker(text, thinker, tokenizer, capture, probe, device="cuda"):
    """Run text through Thinker → get response + emotion."""
    chat_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    tokens = tokenizer(chat_text, return_tensors="pt").to(device)

    with torch.no_grad():
        # Forward pass for hidden states
        capture.clear()
        thinker(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"],
                output_hidden_states=True)

        # Emotion probe
        delta_f32 = {k: v.float() for k, v in capture.deltanet_states.items()}
        attn_f32 = {k: v.float() for k, v in capture.attention_states.items()}
        probe_out = probe(delta_f32, attn_f32)

        # Generate text response
        gen_output = thinker.generate(
            **tokens, max_new_tokens=100, do_sample=True,
            temperature=0.7, top_p=0.9, repetition_penalty=1.1,
        )
    gen_tokens = gen_output[0][tokens["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    return {
        "response": response,
        "emotion_label": probe_out.get("emotion_label", "neutral"),
        "emotion_probs": probe_out.get("emotion_probs"),
        "prosody": probe_out.get("prosody_labels", {}),
        "conditioning_vector": probe_out.get("conditioning_vector"),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=None)
    parser.add_argument("--voice", default=None, help="Reference speaker .wav")
    parser.add_argument("--lora-path", default="checkpoints/living-agent/lora")
    parser.add_argument("--probe-ckpt", default="checkpoints/probe/probe_best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-turbo", action="store_true", help="Use full Chatterbox (has emotion control)")
    args = parser.parse_args()

    # Find a voice
    voice = args.voice
    if not voice:
        import glob
        voices = glob.glob("data/reference_speakers/*.wav")
        if voices:
            voice = voices[0]
        else:
            print("ERROR: No reference speaker. Provide --voice path/to/speaker.wav")
            sys.exit(1)

    # Load Thinker + Probe
    thinker, tokenizer, capture, probe = load_thinker_and_probe(
        args.lora_path, args.probe_ckpt, args.device
    )

    # Create connector
    from src.model.connector import ThinkerTalkerConnector
    use_turbo = not args.no_turbo
    connector = ThinkerTalkerConnector(
        voice_path=voice,
        device=args.device,
        use_turbo=use_turbo,
    )

    test_prompts = [args.text] if args.text else [
        "I'm so happy to see you today!",
        "This makes me absolutely furious.",
        "The weather is nice today.",
        "Oh wow, I can't believe that just happened!",
        "I'm feeling really calm and peaceful.",
    ]

    os.makedirs("test_output", exist_ok=True)

    print(f"\nUsing {'Turbo' if use_turbo else 'Full'} Chatterbox")
    print(f"Voice: {voice}")
    print(f"{'='*60}\n")

    for i, prompt in enumerate(test_prompts):
        print(f"[{i+1}/{len(test_prompts)}] Prompt: \"{prompt}\"")

        # Step 1: Thinker generates response + emotion
        result = run_thinker(prompt, thinker, tokenizer, capture, probe, args.device)
        print(f"  Emotion: {result['emotion_label']}")
        print(f"  Prosody: {result['prosody']}")
        print(f"  Response: \"{result['response'][:80]}{'...' if len(result['response']) > 80 else ''}\"")

        # Step 2: Connector cleans text and maps style → Chatterbox generates audio
        style = connector.map_style(
            result["emotion_label"],
            result["prosody"],
            result["conditioning_vector"],
        )
        print(f"  Style: exagg={style['exaggeration']:.2f}, cfg={style['cfg_weight']:.2f}, temp={style['temperature']:.2f}")

        wav = connector.generate(
            text=result["response"],
            emotion_label=result["emotion_label"],
            prosody=result["prosody"],
            conditioning_vector=result["conditioning_vector"],
        )

        # Save audio
        if isinstance(wav, torch.Tensor):
            wav_np = wav.cpu().numpy()
        else:
            wav_np = np.array(wav)
        if wav_np.ndim == 2:
            wav_np = wav_np.squeeze(0)
        wav_np = np.asarray(wav_np, dtype=np.float32)

        out_path = f"test_output/sample_{i:02d}_{result['emotion_label']}.wav"
        sf.write(out_path, wav_np, connector.sr)
        print(f"  Audio: {out_path} ({len(wav_np)/connector.sr:.1f}s)")
        print()

    # Free Thinker VRAM (Chatterbox uses it now)
    del thinker
    torch.cuda.empty_cache()

    print(f"{'='*60}")
    print(f"Done! {len(test_prompts)} samples saved to test_output/")
    print(f"Listen to them to evaluate quality.")


if __name__ == "__main__":
    main()
