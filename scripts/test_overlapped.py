#!/usr/bin/env python3
"""Test overlapped streaming: Thinker text streams while TTS generates audio.

Compares three modes:
  1. Non-overlapped: think ALL → then speak ALL (baseline)
  2. Overlapped: Thinker streams text → TTS starts on first clause immediately
  3. Full loop test: TTS → Whisper → verify (optional, --loop flag)

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/test_overlapped.py
    CUDA_VISIBLE_DEVICES=1 python scripts/test_overlapped.py --text "Hello!"
    CUDA_VISIBLE_DEVICES=1 python scripts/test_overlapped.py --compare  # side-by-side timing
    CUDA_VISIBLE_DEVICES=1 python scripts/test_overlapped.py --loop     # full TTS→STT loop test
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

import numpy as np
import soundfile as sf


def test_overlapped(text, voice_path, lora_path, probe_ckpt, device, chunk_size):
    """Test overlapped streaming pipeline."""
    from src.inference.streaming import OverlappedStreamingPipeline

    print("Loading OverlappedStreamingPipeline...")
    pipeline = OverlappedStreamingPipeline(
        lora_path=lora_path,
        probe_ckpt=probe_ckpt,
        voice_path=voice_path,
        device=device,
        chunk_size=chunk_size,
    )

    print(f"\nOverlapped streaming: \"{text}\"\n")

    all_audio = []
    t0 = time.time()
    first_chunk_time = None

    for chunk in pipeline.stream(text):
        elapsed = time.time() - t0
        if first_chunk_time is None:
            first_chunk_time = elapsed

        duration = len(chunk.audio) / chunk.sample_rate
        all_audio.append(chunk.audio)

        status = "FINAL" if chunk.is_final else "chunk"
        print(f"  [{elapsed:.2f}s] {status}: {duration:.2f}s audio "
              f"({chunk.total_tokens} total tokens)")

    total_time = time.time() - t0
    total_audio = np.concatenate(all_audio) if all_audio else np.array([])
    sr = pipeline.stream_tts.tts.sr
    total_duration = len(total_audio) / sr if len(total_audio) > 0 else 0

    print(f"\n--- Overlapped Results ---")
    print(f"  First audio latency: {first_chunk_time:.2f}s" if first_chunk_time else "  No audio")
    print(f"  Total generation:    {total_time:.2f}s")
    print(f"  Total audio:         {total_duration:.2f}s")
    if total_duration > 0:
        print(f"  Realtime factor:     {total_time/total_duration:.2f}x")

    os.makedirs("test_output", exist_ok=True)
    out_path = "test_output/overlapped_output.wav"
    if len(total_audio) > 0:
        sf.write(out_path, total_audio, sr)
        print(f"  Saved: {out_path}")

    return first_chunk_time, total_time, total_duration


def test_non_overlapped(text, voice_path, lora_path, probe_ckpt, device, chunk_size):
    """Test non-overlapped baseline (think ALL then speak ALL)."""
    from src.inference.streaming import StreamingConnectorPipeline

    print("Loading StreamingConnectorPipeline (non-overlapped)...")
    pipeline = StreamingConnectorPipeline(
        lora_path=lora_path,
        probe_ckpt=probe_ckpt,
        voice_path=voice_path,
        device=device,
        chunk_size=chunk_size,
    )

    print(f"\nNon-overlapped: \"{text}\"\n")

    all_audio = []
    t0 = time.time()
    first_chunk_time = None

    for chunk in pipeline.speak(text):
        elapsed = time.time() - t0
        if first_chunk_time is None:
            first_chunk_time = elapsed

        duration = len(chunk.audio) / chunk.sample_rate
        all_audio.append(chunk.audio)

    total_time = time.time() - t0
    total_audio = np.concatenate(all_audio) if all_audio else np.array([])
    sr = pipeline.stream_tts.tts.sr
    total_duration = len(total_audio) / sr if len(total_audio) > 0 else 0

    print(f"\n--- Non-Overlapped Results ---")
    print(f"  First audio latency: {first_chunk_time:.2f}s" if first_chunk_time else "  No audio")
    print(f"  Total generation:    {total_time:.2f}s")
    print(f"  Total audio:         {total_duration:.2f}s")

    os.makedirs("test_output", exist_ok=True)
    out_path = "test_output/non_overlapped_output.wav"
    if len(total_audio) > 0:
        sf.write(out_path, total_audio, sr)
        print(f"  Saved: {out_path}")

    return first_chunk_time, total_time, total_duration


def test_loop(voice_path, lora_path, probe_ckpt, device, chunk_size):
    """Full loop test: TTS generates "user speech" → Whisper transcribes → verify.

    This tests the entire stack end-to-end:
      1. Generate a test prompt as audio (simulated user)
      2. Transcribe it with Whisper (STT)
      3. Feed transcription to Thinker (thinking)
      4. Stream response through TTS (speaking)
      5. Transcribe the response with Whisper (verification)
      6. Check if the loop produced coherent results
    """
    from src.inference.streaming import OverlappedStreamingPipeline

    print("=" * 60)
    print("FULL LOOP TEST: TTS → Whisper → Thinker → TTS → Whisper")
    print("=" * 60)

    # Load pipeline
    print("\nLoading pipeline...")
    pipeline = OverlappedStreamingPipeline(
        lora_path=lora_path,
        probe_ckpt=probe_ckpt,
        voice_path=voice_path,
        device=device,
        chunk_size=chunk_size,
    )

    # Load Whisper for STT
    print("Loading Whisper...")
    import whisper
    whisper_model = whisper.load_model("base", device=device)

    os.makedirs("test_output", exist_ok=True)

    # Step 1: Generate "user speech" using TTS
    user_prompt = "What is the meaning of life?"
    print(f"\n[Step 1] Generating user speech: \"{user_prompt}\"")

    user_audio = []
    for chunk in pipeline.stream_tts.stream(user_prompt, chunk_size=chunk_size):
        user_audio.append(chunk.audio)

    user_wav = np.concatenate(user_audio) if user_audio else np.array([])
    sr = pipeline.stream_tts.tts.sr

    user_wav_path = "test_output/loop_user_speech.wav"
    sf.write(user_wav_path, user_wav, sr)
    print(f"  Saved user speech: {user_wav_path} ({len(user_wav)/sr:.1f}s)")

    # Step 2: Transcribe user speech with Whisper
    print(f"\n[Step 2] Transcribing user speech with Whisper...")
    result = whisper_model.transcribe(user_wav_path)
    transcribed_input = result["text"].strip()
    print(f"  Whisper heard: \"{transcribed_input}\"")

    # Step 3: Feed to Thinker → stream TTS response
    print(f"\n[Step 3] Thinker responding to: \"{transcribed_input}\"")
    t0 = time.time()

    response_audio = []
    for chunk in pipeline.stream(transcribed_input):
        response_audio.append(chunk.audio)

    response_wav = np.concatenate(response_audio) if response_audio else np.array([])
    think_speak_time = time.time() - t0

    response_wav_path = "test_output/loop_response_speech.wav"
    if len(response_wav) > 0:
        sf.write(response_wav_path, response_wav, sr)
        print(f"  Saved response: {response_wav_path} ({len(response_wav)/sr:.1f}s)")
        print(f"  Think+speak time: {think_speak_time:.2f}s")

    # Step 4: Transcribe response with Whisper
    print(f"\n[Step 4] Transcribing AI response with Whisper...")
    if len(response_wav) > 0:
        result = whisper_model.transcribe(response_wav_path)
        transcribed_output = result["text"].strip()
        print(f"  Whisper heard: \"{transcribed_output}\"")
    else:
        transcribed_output = ""
        print("  No audio to transcribe!")

    # Step 5: Evaluate
    print(f"\n{'=' * 60}")
    print("LOOP TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"  Original prompt:      \"{user_prompt}\"")
    print(f"  Whisper transcribed:  \"{transcribed_input}\"")
    print(f"  AI spoken response:   \"{transcribed_output}\"")
    print(f"  Response length:      {len(transcribed_output)} chars")
    print(f"  Total loop time:      {think_speak_time:.2f}s")

    # Basic coherence check
    if len(transcribed_output) > 10:
        print(f"  Status: PASS (got coherent response)")
    else:
        print(f"  Status: WEAK (response too short, may need tuning)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="I'm really excited to show you how overlapped streaming works! It should start speaking much faster than before.")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--lora-path", default="checkpoints/living-agent/lora")
    parser.add_argument("--probe-ckpt", default="checkpoints/probe/probe_best.pt")
    parser.add_argument("--compare", action="store_true", help="Compare overlapped vs non-overlapped timing")
    parser.add_argument("--loop", action="store_true", help="Full loop test: TTS→Whisper→Thinker→TTS→Whisper")
    args = parser.parse_args()

    # Find voice
    voice = args.voice
    if not voice:
        import glob
        voices = glob.glob("data/reference_speakers/*.wav")
        if voices:
            voice = voices[0]
        else:
            print("ERROR: No reference speaker. Provide --voice")
            sys.exit(1)

    if args.loop:
        test_loop(voice, args.lora_path, args.probe_ckpt, args.device, args.chunk_size)
    elif args.compare:
        print("=" * 60)
        print("COMPARISON: Overlapped vs Non-Overlapped")
        print("=" * 60)

        ol_first, ol_total, ol_dur = test_overlapped(
            args.text, voice, args.lora_path, args.probe_ckpt, args.device, args.chunk_size,
        )

        print(f"\n{'=' * 60}\n")

        # Note: non-overlapped loads models again (separate pipeline)
        # In practice they'd share, but for fair comparison we time each
        nol_first, nol_total, nol_dur = test_non_overlapped(
            args.text, voice, args.lora_path, args.probe_ckpt, args.device, args.chunk_size,
        )

        print(f"\n{'=' * 60}")
        print("COMPARISON RESULTS")
        print(f"{'=' * 60}")
        if ol_first and nol_first:
            print(f"  First audio latency:")
            print(f"    Overlapped:     {ol_first:.2f}s")
            print(f"    Non-overlapped: {nol_first:.2f}s")
            print(f"    Speedup:        {nol_first/ol_first:.1f}x faster")
        print(f"  Total time:")
        print(f"    Overlapped:     {ol_total:.2f}s")
        print(f"    Non-overlapped: {nol_total:.2f}s")
    else:
        test_overlapped(
            args.text, voice, args.lora_path, args.probe_ckpt, args.device, args.chunk_size,
        )


if __name__ == "__main__":
    main()
