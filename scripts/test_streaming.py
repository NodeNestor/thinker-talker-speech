#!/usr/bin/env python3
"""Test streaming speech generation — audio plays as it's generated.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/test_streaming.py
    CUDA_VISIBLE_DEVICES=1 python scripts/test_streaming.py --text "Hello!"
    CUDA_VISIBLE_DEVICES=1 python scripts/test_streaming.py --chunk-size 30  # lower latency
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


def test_streaming_tts(text, voice_path, device, chunk_size):
    """Test just the streaming TTS (no Thinker)."""
    from src.inference.streaming import StreamingTTSPipeline

    print("Loading Chatterbox Turbo (streaming mode)...")
    pipeline = StreamingTTSPipeline.from_pretrained(device=device)
    pipeline.set_voice(voice_path)

    print(f"\nStreaming: \"{text}\"")
    print(f"Chunk size: {chunk_size} tokens\n")

    all_audio = []
    t0 = time.time()
    first_chunk_time = None

    for chunk in pipeline.stream(text, chunk_size=chunk_size):
        elapsed = time.time() - t0
        if first_chunk_time is None:
            first_chunk_time = elapsed

        duration = len(chunk.audio) / chunk.sample_rate
        all_audio.append(chunk.audio)

        status = "FINAL" if chunk.is_final else f"chunk"
        print(f"  [{elapsed:.2f}s] {status}: {duration:.2f}s audio "
              f"({chunk.token_count} tokens, {chunk.total_tokens} total)")

    total_time = time.time() - t0
    total_audio = np.concatenate(all_audio) if all_audio else np.array([])
    total_duration = len(total_audio) / pipeline.tts.sr if len(total_audio) > 0 else 0

    print(f"\n--- Results ---")
    print(f"  First chunk latency: {first_chunk_time:.2f}s")
    print(f"  Total generation:    {total_time:.2f}s")
    print(f"  Total audio:         {total_duration:.2f}s")
    print(f"  Realtime factor:     {total_time/total_duration:.2f}x" if total_duration > 0 else "")

    # Save combined audio
    os.makedirs("test_output", exist_ok=True)
    out_path = "test_output/streaming_output.wav"
    sf.write(out_path, total_audio, pipeline.tts.sr)
    print(f"  Saved: {out_path}")

    return first_chunk_time, total_time, total_duration


def test_full_pipeline(text, voice_path, lora_path, probe_ckpt, device, chunk_size):
    """Test full pipeline: Thinker → emotion → streaming Chatterbox."""
    from src.inference.streaming import StreamingConnectorPipeline

    print("Loading full pipeline (Thinker + Probe + Chatterbox)...")
    pipeline = StreamingConnectorPipeline(
        lora_path=lora_path,
        probe_ckpt=probe_ckpt,
        voice_path=voice_path,
        device=device,
        chunk_size=chunk_size,
    )

    print(f"\nFull pipeline: \"{text}\"\n")

    all_audio = []
    t0 = time.time()
    think_time = None
    first_audio_time = None

    # Think first (this is the non-streaming part)
    result = pipeline.think(text)
    think_time = time.time() - t0
    print(f"  Think time: {think_time:.2f}s")
    print(f"  Response: \"{result['response'][:80]}...\"")

    # Stream audio
    clean_text = pipeline.connector.clean_text(result["response"])
    style = pipeline.connector.map_style(
        result["emotion_label"], result["prosody"], result["conditioning_vector"]
    )

    for chunk in pipeline.stream_tts.stream(
        text=clean_text,
        temperature=style["temperature"],
        chunk_size=chunk_size,
    ):
        elapsed = time.time() - t0
        if first_audio_time is None:
            first_audio_time = elapsed

        duration = len(chunk.audio) / chunk.sample_rate
        all_audio.append(chunk.audio)

        status = "FINAL" if chunk.is_final else "chunk"
        print(f"  [{elapsed:.2f}s] {status}: {duration:.2f}s audio")

    total_time = time.time() - t0
    total_audio = np.concatenate(all_audio) if all_audio else np.array([])
    total_duration = len(total_audio) / pipeline.stream_tts.tts.sr if len(total_audio) > 0 else 0

    print(f"\n--- Full Pipeline Results ---")
    print(f"  Think time:          {think_time:.2f}s")
    print(f"  First audio latency: {first_audio_time:.2f}s" if first_audio_time else "  No audio generated")
    print(f"  Total time:          {total_time:.2f}s")
    print(f"  Total audio:         {total_duration:.2f}s")

    os.makedirs("test_output", exist_ok=True)
    out_path = "test_output/full_pipeline_streaming.wav"
    if len(total_audio) > 0:
        sf.write(out_path, total_audio, pipeline.stream_tts.tts.sr)
        print(f"  Saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="I'm really excited to show you how streaming works!")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--full", action="store_true", help="Test full Thinker→Chatterbox pipeline")
    parser.add_argument("--lora-path", default="checkpoints/living-agent/lora")
    parser.add_argument("--probe-ckpt", default="checkpoints/probe/probe_best.pt")
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

    if args.full:
        test_full_pipeline(
            args.text, voice, args.lora_path, args.probe_ckpt,
            args.device, args.chunk_size,
        )
    else:
        test_streaming_tts(args.text, voice, args.device, args.chunk_size)


if __name__ == "__main__":
    main()
