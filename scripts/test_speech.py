#!/usr/bin/env python3
"""Test speech pipeline: TTS (Chatterbox Turbo) + STT (Whisper).

Quick test to verify:
  1. Chatterbox Turbo generates speech from text
  2. Whisper transcribes it back
  3. Round-trip works (text -> speech -> text)
"""

import torch
import soundfile as sf
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _save_wav(path, wav, sr):
    """Save wav tensor to file using soundfile."""
    import numpy as np
    audio = wav.squeeze().cpu().numpy()
    if audio.ndim > 1:
        audio = audio[0]
    sf.write(path, audio, sr)


def test_tts():
    """Test Chatterbox Turbo TTS."""
    print("=" * 60)
    print("Test 1: Chatterbox Turbo TTS")
    print("=" * 60)

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    print("  Loading model...")
    t0 = time.time()
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Sample rate: {model.sr}")

    # Test basic generation (no voice cloning — uses default voice)
    text = "Hello! I'm your AI assistant. How can I help you today?"
    print(f"  Generating: \"{text}\"")
    t0 = time.time()
    wav = model.generate(text)
    gen_time = time.time() - t0
    duration = wav.shape[-1] / model.sr
    print(f"  Generated {duration:.2f}s audio in {gen_time:.2f}s (RTF: {gen_time/duration:.2f}x)")

    # Save
    os.makedirs("test_output", exist_ok=True)
    _save_wav("test_output/tts_basic.wav", wav, model.sr)
    print(f"  Saved to test_output/tts_basic.wav")

    # Test with paralinguistic tags (matching our training format)
    text2 = "Oh wow, that's amazing! [laugh] I didn't expect that at all. [pause] Let me think about it."
    print(f"\n  Generating with tags: \"{text2}\"")
    t0 = time.time()
    wav2 = model.generate(text2)
    gen_time = time.time() - t0
    duration2 = wav2.shape[-1] / model.sr
    print(f"  Generated {duration2:.2f}s audio in {gen_time:.2f}s (RTF: {gen_time/duration2:.2f}x)")

    _save_wav("test_output/tts_with_tags.wav", wav2, model.sr)
    print(f"  Saved to test_output/tts_with_tags.wav")

    print("  PASS\n")
    return model, wav


def test_stt(audio_path="test_output/tts_basic.wav"):
    """Test Whisper STT."""
    print("=" * 60)
    print("Test 2: Whisper STT")
    print("=" * 60)

    import whisper

    print("  Loading whisper-small...")
    t0 = time.time()
    model = whisper.load_model("small", device="cuda")
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    print(f"  Transcribing: {audio_path}")
    t0 = time.time()
    # Load audio with soundfile (avoids ffmpeg dependency)
    import numpy as np
    audio_data, file_sr = sf.read(audio_path, dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    # Resample to 16kHz if needed (whisper expects 16kHz)
    if file_sr != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=file_sr, target_sr=16000)
    result = model.transcribe(audio_data)
    trans_time = time.time() - t0
    print(f"  Transcription ({trans_time:.2f}s): \"{result['text'].strip()}\"")
    print(f"  Language: {result['language']}")

    print("  PASS\n")
    return result


def test_roundtrip():
    """Test full round-trip: text -> TTS -> STT -> text."""
    print("=" * 60)
    print("Test 3: Round-trip (text -> speech -> text)")
    print("=" * 60)

    from chatterbox.tts_turbo import ChatterboxTurboTTS
    import whisper

    tts = ChatterboxTurboTTS.from_pretrained(device="cuda")
    stt = whisper.load_model("small", device="cuda")

    test_phrases = [
        "Please open my email and check for new messages.",
        "What's the weather like in Stockholm today?",
        "Can you set a timer for fifteen minutes?",
    ]

    for phrase in test_phrases:
        # TTS
        wav = tts.generate(phrase)
        _save_wav("test_output/_roundtrip_temp.wav", wav, tts.sr)

        # STT — load with soundfile, resample to 16kHz
        import numpy as np
        import librosa
        audio_data, file_sr = sf.read("test_output/_roundtrip_temp.wav", dtype="float32")
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        if file_sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=file_sr, target_sr=16000)
        result = stt.transcribe(audio_data)
        transcribed = result["text"].strip()

        match = phrase.lower().rstrip(".!?") in transcribed.lower() or \
                transcribed.lower().rstrip(".!?") in phrase.lower()
        status = "OK" if match else "MISMATCH"
        print(f"  [{status}] \"{phrase}\"")
        print(f"         -> \"{transcribed}\"")

    # Cleanup
    try:
        os.remove("test_output/_roundtrip_temp.wav")
    except OSError:
        pass

    print("\n  PASS\n")


def test_microphone():
    """Test live microphone input -> Whisper."""
    print("=" * 60)
    print("Test 4: Live microphone (5 seconds)")
    print("=" * 60)

    try:
        import sounddevice as sd
    except ImportError:
        print("  SKIP: sounddevice not installed (pip install sounddevice)")
        return

    import whisper
    import numpy as np

    stt = whisper.load_model("small", device="cuda")

    sr = 16000
    duration = 5
    print(f"  Recording {duration}s from microphone (speak now!)...")
    audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("  Recording done.")

    # Save recording
    audio_tensor = torch.from_numpy(audio.T)  # [1, samples]
    _save_wav("test_output/mic_recording.wav", audio_tensor, sr)

    # Transcribe
    result = stt.transcribe("test_output/mic_recording.wav")
    print(f"  Transcribed: \"{result['text'].strip()}\"")
    print("  PASS\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["tts", "stt", "roundtrip", "mic", "all"],
                        default="all", help="Which test to run")
    args = parser.parse_args()

    if args.test in ("tts", "all"):
        test_tts()
    if args.test in ("stt", "all"):
        test_stt()
    if args.test in ("roundtrip", "all"):
        test_roundtrip()
    if args.test in ("mic", "all"):
        test_microphone()

    print("All tests done!")
