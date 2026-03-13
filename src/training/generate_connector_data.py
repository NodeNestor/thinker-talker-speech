"""Generate synthetic training data for the Thinker→Talker connector.

Instead of relying solely on LibriSpeech, we use our own models:
  1. Thinker generates diverse text (varied emotions, topics, styles)
  2. Chatterbox Turbo synthesizes speech from that text
  3. Result: perfectly paired (text, audio) with unlimited diversity

This gives us much better coverage than a fixed dataset, especially for
emotion and speaker diversity.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# Emotional text prompts to seed the Thinker
EMOTION_PROMPTS = {
    "happy": [
        "Tell me something exciting that happened today!",
        "What's the best news you've heard recently?",
        "Describe something that makes you laugh every time.",
        "What are you most grateful for right now?",
    ],
    "sad": [
        "Tell me about a time you felt really down.",
        "What's something that makes you feel melancholy?",
        "Describe a bittersweet memory.",
        "What do you miss most about the past?",
    ],
    "angry": [
        "What really frustrates you about the world?",
        "Tell me about something that's completely unfair.",
        "What makes you absolutely furious?",
        "Describe a time someone really let you down.",
    ],
    "excited": [
        "What are you most looking forward to?",
        "Tell me about your biggest dream coming true!",
        "What's the most thrilling thing you've ever done?",
        "Describe the moment you realized you won!",
    ],
    "calm": [
        "Describe a peaceful evening at home.",
        "What helps you relax after a long day?",
        "Tell me about your favorite quiet place.",
        "Walk me through your morning routine.",
    ],
    "fearful": [
        "What's your biggest fear?",
        "Describe a time you were really scared.",
        "What keeps you up at night worrying?",
        "Tell me about a close call you had.",
    ],
    "surprised": [
        "Tell me about the most unexpected thing that ever happened to you.",
        "What's something that completely blew your mind?",
        "Describe a plot twist in your life.",
        "What's the weirdest coincidence you've experienced?",
    ],
    "neutral": [
        "Explain how a coffee machine works.",
        "Describe the layout of your neighborhood.",
        "Tell me about the weather this week.",
        "What did you have for lunch today?",
    ],
}

# Reference audio files for speaker diversity (will scan for .wav/.flac)
DEFAULT_REFERENCE_DIR = "data/reference_speakers"


def generate_texts(thinker, tokenizer, num_per_emotion=50, max_new_tokens=80, device="cuda"):
    """Generate diverse text responses using the Thinker."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(thinker)

    generated = []
    for emotion, prompts in EMOTION_PROMPTS.items():
        print(f"  Generating {num_per_emotion} '{emotion}' texts...")
        for i in range(num_per_emotion):
            prompt = prompts[i % len(prompts)]
            # Format as chat
            chat_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(chat_text, return_tensors="pt").to(device)

            with torch.no_grad():
                output = thinker.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    repetition_penalty=1.1,
                )
            # Decode only the generated part
            gen_tokens = output[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            if len(text) > 10:  # skip empty/tiny generations
                generated.append({
                    "text": text,
                    "emotion": emotion,
                    "prompt": prompt,
                })

    print(f"  Generated {len(generated)} texts total")
    return generated


def synthesize_audio(tts_model, texts, reference_audios, output_dir, sr=24000):
    """Synthesize speech from text using Chatterbox Turbo.

    Each text is synthesized with a random reference speaker for diversity.
    """
    import soundfile as sf

    os.makedirs(output_dir, exist_ok=True)
    samples = []

    for i, item in enumerate(tqdm(texts, desc="Synthesizing audio")):
        text = item["text"]
        # Pick a random reference speaker
        ref_audio = reference_audios[i % len(reference_audios)]

        try:
            wav = tts_model.generate(text, audio_prompt_path=ref_audio)
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            if isinstance(wav, np.ndarray) and wav.ndim == 2:
                wav = wav.squeeze(0)
            wav = np.asarray(wav, dtype=np.float32)

            # Save audio (soundfile — no TorchCodec dependency)
            audio_path = os.path.join(output_dir, f"sample_{i:05d}.wav")
            sf.write(audio_path, wav, sr)

            samples.append({
                **item,
                "audio_path": audio_path,
                "speaker_ref": ref_audio,
                "sample_rate": sr,
            })
        except Exception as e:
            print(f"  Failed to synthesize sample {i}: {e}")
            continue

    return samples


def find_reference_audios(ref_dir=DEFAULT_REFERENCE_DIR, librispeech_dir=None):
    """Find reference audio files for speaker diversity.

    First checks ref_dir for curated references. Falls back to
    extracting diverse speakers from LibriSpeech.
    """
    refs = []

    # Check curated reference dir
    if os.path.isdir(ref_dir):
        for ext in ("*.wav", "*.flac", "*.mp3"):
            refs.extend(str(p) for p in Path(ref_dir).glob(ext))

    if refs:
        print(f"  Found {len(refs)} reference speakers in {ref_dir}")
        return refs

    # Fall back: extract from LibriSpeech cache
    print(f"  No reference audio in {ref_dir}, extracting from LibriSpeech...")
    from datasets import load_dataset, Audio
    import soundfile as sf
    import io

    ds = load_dataset("librispeech_asr", "clean", split="train.100")
    ds = ds.cast_column("audio", Audio(decode=False))

    # Collect multiple utterances per speaker to ensure >= 6s clips
    os.makedirs(ref_dir, exist_ok=True)
    speaker_audio = {}  # spk_id -> list of audio arrays
    MIN_DURATION = 6.0  # seconds — Chatterbox requires > 5s

    for item in ds:
        spk = item["speaker_id"]
        if spk in speaker_audio and len(speaker_audio) >= 50:
            # Already have enough speakers, skip new ones
            if spk not in speaker_audio:
                continue

        audio_info = item["audio"]
        if audio_info.get("bytes"):
            audio, sr = sf.read(io.BytesIO(audio_info["bytes"]), dtype="float32")
        else:
            audio, sr = sf.read(audio_info["path"], dtype="float32")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio = audio.astype(np.float32)

        if spk not in speaker_audio:
            speaker_audio[spk] = []
        speaker_audio[spk].append(audio)

        # Check if all speakers have enough audio
        if len(speaker_audio) >= 50:
            all_long = all(
                sum(len(a) for a in chunks) / 16000 >= MIN_DURATION
                for chunks in speaker_audio.values()
            )
            if all_long:
                break

    # Concatenate and save (cap at 10s — Chatterbox only needs ~6s)
    MAX_DURATION = 10.0
    for spk, chunks in speaker_audio.items():
        combined = np.concatenate(chunks).astype(np.float32)
        duration = len(combined) / 16000
        if duration < MIN_DURATION:
            continue  # skip speakers without enough audio
        if duration > MAX_DURATION:
            combined = combined[:int(MAX_DURATION * 16000)]

        ref_path = os.path.join(ref_dir, f"speaker_{spk}.wav")
        sf.write(ref_path, combined, 16000, subtype="PCM_16")
        refs.append(ref_path)

    print(f"  Extracted {len(refs)} reference speakers from LibriSpeech")
    return refs


def generate_dataset(
    lora_path="checkpoints/living-agent/lora",
    output_dir="data/synthetic_connector",
    num_per_emotion=50,
    device="cuda",
):
    """Full pipeline: generate text → synthesize audio → save dataset."""

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load Thinker
    print("Loading Thinker...")
    import unsloth
    from unsloth import FastLanguageModel
    thinker, processor = FastLanguageModel.from_pretrained(
        lora_path, device_map=device, dtype=torch.bfloat16,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Step 2: Generate diverse texts
    print(f"\nGenerating {num_per_emotion} texts per emotion ({len(EMOTION_PROMPTS)} emotions)...")
    texts = generate_texts(thinker, tokenizer, num_per_emotion=num_per_emotion, device=device)

    # Free Thinker from VRAM before loading Chatterbox
    del thinker
    torch.cuda.empty_cache()
    print("  Freed Thinker from VRAM")

    # Step 3: Find reference speaker audio
    print("\nFinding reference speakers...")
    ref_audios = find_reference_audios()

    # Step 4: Load Chatterbox and synthesize
    print("\nLoading Chatterbox Turbo...")
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    tts = ChatterboxTurboTTS.from_pretrained(device=device)
    sr = tts.sr

    audio_dir = os.path.join(output_dir, "audio")
    print(f"\nSynthesizing {len(texts)} audio samples...")
    samples = synthesize_audio(tts, texts, ref_audios, audio_dir, sr=sr)

    del tts
    torch.cuda.empty_cache()

    # Step 5: Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "samples": samples,
            "num_samples": len(samples),
            "sample_rate": sr,
            "emotions": list(EMOTION_PROMPTS.keys()),
            "num_speakers": len(ref_audios),
        }, f, indent=2)

    print(f"\nDone! Generated {len(samples)} samples")
    print(f"Saved to {output_dir}/")
    print(f"Manifest: {manifest_path}")
    return samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic connector training data")
    parser.add_argument("--lora-path", default="checkpoints/living-agent/lora")
    parser.add_argument("--output", default="data/synthetic_connector")
    parser.add_argument("--num-per-emotion", type=int, default=50,
                        help="Number of texts to generate per emotion category")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    generate_dataset(
        lora_path=args.lora_path,
        output_dir=args.output,
        num_per_emotion=args.num_per_emotion,
        device=args.device,
    )
