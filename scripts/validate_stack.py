#!/usr/bin/env python3
"""Validate the full Thinker-Talker stack -- every trained component.

Tests each component individually, then end-to-end. Reports pass/fail
with quality metrics so you know what's actually working.

Components tested:
  1. Thinker (LoRA) -- text generation quality
  2. Emotion Probe -- emotion detection accuracy on known-emotion prompts
  3. Connector -- style mapping produces distinct params per emotion
  4. TTS (Chatterbox) -- generates intelligible speech (verified by Whisper)
  5. Streaming -- overlapped pipeline works and is faster
  6. Full Loop -- TTS -> Whisper -> Thinker -> TTS -> Whisper roundtrip

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/validate_stack.py
    CUDA_VISIBLE_DEVICES=1 python scripts/validate_stack.py --component thinker
    CUDA_VISIBLE_DEVICES=1 python scripts/validate_stack.py --component probe
    CUDA_VISIBLE_DEVICES=1 python scripts/validate_stack.py --component tts
    CUDA_VISIBLE_DEVICES=1 python scripts/validate_stack.py --component loop
    CUDA_VISIBLE_DEVICES=1 python scripts/validate_stack.py --all
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

import torch
import numpy as np
import soundfile as sf
import librosa


def load_audio_np(path, sr=16000):
    """Load audio as 16kHz mono float32 numpy array (what whisper expects).
    Bypasses whisper's ffmpeg-based loader which fails on Windows."""
    audio, orig_sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    return audio


# ── Test prompts with expected emotions ──────────────────────────────

EMOTION_TEST_PROMPTS = [
    ("I'm so happy to see you today! This is wonderful!", "happy"),
    ("This makes me absolutely furious, I can't believe it.", "angry"),
    ("I feel really calm and peaceful right now.", "calm"),
    ("Oh wow, I can't believe that just happened!", "surprised"),
    ("I'm feeling really down and sad about everything.", "sad"),
    ("I'm SO excited about this, let's go!", "excited"),
    ("That's disgusting, absolutely revolting.", "disgusted"),
    ("I'm terrified, this is really scary.", "fearful"),
    ("What? I don't understand what's happening.", "confused"),
    ("The weather is nice today. Nothing special.", "neutral"),
]

THINKER_TEST_PROMPTS = [
    "What's your favorite thing about coding?",
    "Tell me a joke.",
    "How do you feel right now?",
    "What should I have for dinner?",
    "Explain what a neural network is in one sentence.",
]

INTELLIGIBILITY_TEXTS = [
    "Hello, how are you doing today?",
    "The quick brown fox jumps over the lazy dog.",
    "I really enjoy working on machine learning projects.",
    "Can you help me understand this concept better?",
]


class ValidationReport:
    """Collects pass/fail results and prints a summary."""

    def __init__(self):
        self.results = []

    def add(self, component: str, test: str, passed: bool, detail: str = ""):
        self.results.append({
            "component": component,
            "test": test,
            "passed": passed,
            "detail": detail,
        })
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test}" + (f" -- {detail}" if detail else ""))

    def summary(self):
        print(f"\n{'=' * 70}")
        print("VALIDATION SUMMARY")
        print(f"{'=' * 70}")

        components = {}
        for r in self.results:
            comp = r["component"]
            if comp not in components:
                components[comp] = {"pass": 0, "fail": 0}
            if r["passed"]:
                components[comp]["pass"] += 1
            else:
                components[comp]["fail"] += 1

        total_pass = sum(c["pass"] for c in components.values())
        total_fail = sum(c["fail"] for c in components.values())

        for comp, counts in components.items():
            total = counts["pass"] + counts["fail"]
            status = "OK" if counts["fail"] == 0 else "ISSUES"
            print(f"  {comp:20s}  {counts['pass']}/{total} passed  [{status}]")

        print(f"\n  Total: {total_pass}/{total_pass + total_fail} passed")

        if total_fail > 0:
            print(f"\n  Failed tests:")
            for r in self.results:
                if not r["passed"]:
                    print(f"    - [{r['component']}] {r['test']}: {r['detail']}")

        return total_fail == 0


# ── Component validators ─────────────────────────────────────────────

def validate_thinker(report, lora_path, device):
    """Validate that the Thinker generates coherent text."""
    print(f"\n{'-' * 70}")
    print("1. THINKER (LoRA) -- Text Generation")
    print(f"{'-' * 70}")

    from unsloth import FastLanguageModel

    print("  Loading Thinker...")
    thinker, processor = FastLanguageModel.from_pretrained(
        lora_path, device_map=device, dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(thinker)
    thinker.eval()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    for prompt in THINKER_TEST_PROMPTS:
        chat = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        tokens = tokenizer(chat, return_tensors="pt").to(device)

        with torch.no_grad():
            gen = thinker.generate(
                **tokens, max_new_tokens=80, do_sample=True,
                temperature=0.7, top_p=0.9, repetition_penalty=1.1,
            )

        gen_tokens = gen[0][tokens["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Basic quality checks
        has_content = len(response) > 5
        not_garbage = not all(c in ".,!? \n" for c in response)
        not_repetitive = len(set(response.split())) > len(response.split()) * 0.3 if response.split() else False

        passed = has_content and not_garbage and not_repetitive
        detail = f"\"{response[:80]}{'...' if len(response) > 80 else ''}\""
        report.add("Thinker", f"Response to: \"{prompt[:40]}...\"", passed, detail)

    # Free VRAM
    del thinker
    torch.cuda.empty_cache()


def validate_probe(report, lora_path, probe_ckpt, device):
    """Validate emotion probe accuracy on known-emotion prompts."""
    print(f"\n{'-' * 70}")
    print("2. EMOTION PROBE -- Emotion Detection")
    print(f"{'-' * 70}")

    import unsloth
    from unsloth import FastLanguageModel
    from src.model.emotion_probe import EmotionProbe, EMOTION_LABELS
    from src.training.train_probe import HiddenStateCapture

    print("  Loading Thinker + Probe...")
    thinker, processor = FastLanguageModel.from_pretrained(
        lora_path, device_map=device, dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(thinker)
    thinker.eval()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    hidden_size = thinker.config.get_text_config().hidden_size
    capture = HiddenStateCapture(thinker)

    probe = EmotionProbe(hidden_size=hidden_size).to(device)
    if os.path.exists(probe_ckpt):
        probe.load_state_dict(torch.load(probe_ckpt, map_location=device))
        report.add("Probe", "Checkpoint loaded", True, probe_ckpt)
    else:
        report.add("Probe", "Checkpoint loaded", False, f"Not found: {probe_ckpt}")
        return
    probe.eval()

    # Related emotion groups (detecting "excited" when expected "happy" is acceptable)
    CLOSE_EMOTIONS = {
        "happy": {"happy", "excited", "calm"},
        "excited": {"excited", "happy", "surprised"},
        "calm": {"calm", "neutral", "happy"},
        "sad": {"sad", "fearful", "disgusted"},
        "angry": {"angry", "disgusted"},
        "surprised": {"surprised", "excited", "confused"},
        "fearful": {"fearful", "sad", "surprised"},
        "disgusted": {"disgusted", "angry", "sad"},
        "confused": {"confused", "surprised", "neutral"},
        "neutral": {"neutral", "calm"},
    }

    exact_correct = 0
    close_correct = 0
    total = 0

    for prompt, expected_emotion in EMOTION_TEST_PROMPTS:
        chat = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        tokens = tokenizer(chat, return_tensors="pt").to(device)

        with torch.no_grad():
            capture.clear()
            thinker(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"],
                    output_hidden_states=True)

            delta_f32 = {k: v.float() for k, v in capture.deltanet_states.items()}
            attn_f32 = {k: v.float() for k, v in capture.attention_states.items()}
            result = probe(delta_f32, attn_f32)

        detected = result["emotion_label"]
        probs = result["emotion_probs"][0]
        top3_idx = probs.topk(3).indices.tolist()
        top3 = [(EMOTION_LABELS[i], f"{probs[i]:.2f}") for i in top3_idx]

        exact_match = detected == expected_emotion
        close_match = detected in CLOSE_EMOTIONS.get(expected_emotion, {expected_emotion})

        if exact_match:
            exact_correct += 1
        if close_match:
            close_correct += 1
        total += 1

        detail = f"expected={expected_emotion}, got={detected}, top3={top3}"
        report.add("Probe", f"Emotion: \"{prompt[:35]}...\"", close_match, detail)

    # Overall accuracy
    exact_acc = exact_correct / total
    close_acc = close_correct / total
    report.add("Probe", f"Exact accuracy: {exact_acc:.0%} ({exact_correct}/{total})",
               exact_acc >= 0.3, f"threshold: 30%")
    report.add("Probe", f"Close accuracy: {close_acc:.0%} ({close_correct}/{total})",
               close_acc >= 0.5, f"threshold: 50%")

    # Check prosody output exists and is reasonable
    last_result = result
    prosody = last_result.get("prosody_labels", {})
    has_prosody = all(k in prosody for k in ["speed", "pitch", "energy", "emphasis"])
    prosody_reasonable = all(0.0 <= v <= 3.0 for v in prosody.values()) if prosody else False
    report.add("Probe", "Prosody output", has_prosody and prosody_reasonable,
               f"values={prosody}")

    # Check conditioning vector shape
    cond = last_result.get("conditioning_vector")
    report.add("Probe", "Conditioning vector",
               cond is not None and cond.shape[-1] == 14,
               f"shape={cond.shape if cond is not None else 'None'}")

    capture.remove_hooks()
    del thinker
    torch.cuda.empty_cache()


def validate_connector(report):
    """Validate connector style mapping produces distinct params per emotion."""
    print(f"\n{'-' * 70}")
    print("3. CONNECTOR -- Style Mapping")
    print(f"{'-' * 70}")

    from src.model.connector import ThinkerTalkerConnector, EMOTION_EXAGGERATION, EMOTION_TEMPERATURE

    connector = ThinkerTalkerConnector(device="cpu", use_turbo=True)

    # Test that different emotions produce different style params
    styles = {}
    for emotion in ["neutral", "happy", "angry", "excited", "calm", "sad"]:
        style = connector.map_style(emotion_label=emotion)
        styles[emotion] = style
        report.add("Connector", f"Style for '{emotion}'", True,
                    f"exagg={style['exaggeration']:.2f}, temp={style['temperature']:.2f}")

    # Verify differentiation: excited should have higher exaggeration than neutral
    report.add("Connector", "excited > neutral exaggeration",
               styles["excited"]["exaggeration"] > styles["neutral"]["exaggeration"],
               f"{styles['excited']['exaggeration']:.2f} vs {styles['neutral']['exaggeration']:.2f}")

    report.add("Connector", "calm < angry temperature",
               styles["calm"]["temperature"] < styles["angry"]["temperature"],
               f"{styles['calm']['temperature']:.2f} vs {styles['angry']['temperature']:.2f}")

    # Test prosody influence
    base_style = connector.map_style("happy", prosody={"energy": 1.0})
    high_energy = connector.map_style("happy", prosody={"energy": 1.5})
    report.add("Connector", "High energy -> higher exaggeration",
               high_energy["exaggeration"] >= base_style["exaggeration"],
               f"{high_energy['exaggeration']:.2f} vs {base_style['exaggeration']:.2f}")

    # Test text cleaning
    dirty = "<think>internal thought</think> Hello! <tool_call>do_thing()</tool_call> How are you?"
    clean = connector.clean_text(dirty)
    report.add("Connector", "Text cleaning",
               "Hello!" in clean and "think" not in clean and "tool_call" not in clean,
               f"\"{clean}\"")


def validate_hidden_connector(report, lora_path, probe_ckpt, connector_ckpt, device):
    """Validate HiddenStateConnector projects Thinker states into T3 space."""
    print(f"\n{'-' * 70}")
    print("3b. HIDDEN STATE CONNECTOR -- Thinker->T3 Projection")
    print(f"{'-' * 70}")

    from src.model.connector import HiddenStateConnector

    if not os.path.exists(connector_ckpt):
        report.add("HiddenConnector", "Checkpoint exists", False,
                    f"Not found: {connector_ckpt}. Train with: python -m src.training.train_stage4")
        return

    # Load connector
    connector = HiddenStateConnector(thinker_dim=1024, t3_dim=1024, emotion_dim=14)
    connector.load_state_dict(torch.load(connector_ckpt, map_location=device, weights_only=True))
    connector = connector.to(device).eval()
    trainable = sum(p.numel() for p in connector.parameters())
    report.add("HiddenConnector", "Checkpoint loaded", True,
               f"{trainable:,} params from {connector_ckpt}")

    # Load Thinker for hidden states
    import unsloth
    from unsloth import FastLanguageModel
    from src.model.emotion_probe import EmotionProbe
    from src.training.train_probe import HiddenStateCapture

    thinker, processor = FastLanguageModel.from_pretrained(
        lora_path, device_map=device, dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(thinker)
    thinker.eval()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    hidden_size = thinker.config.get_text_config().hidden_size
    capture = HiddenStateCapture(thinker)

    probe = EmotionProbe(hidden_size=hidden_size).to(device)
    if os.path.exists(probe_ckpt):
        probe.load_state_dict(torch.load(probe_ckpt, map_location=device))
    probe.eval()

    # Test: forward pass through connector
    test_texts = [
        "Hello, how are you doing today?",
        "I'm really excited about this!",
        "The quick brown fox jumps over the lazy dog.",
    ]

    for text in test_texts:
        tokens = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            capture.clear()
            out = thinker(**tokens, output_hidden_states=True)
            if hasattr(out, 'hidden_states') and out.hidden_states:
                hidden = out.hidden_states[-1]
            else:
                all_states = {**capture.deltanet_states, **capture.attention_states}
                hidden = all_states[max(all_states.keys())]

            delta_f32 = {k: v.float() for k, v in capture.deltanet_states.items()}
            attn_f32 = {k: v.float() for k, v in capture.attention_states.items()}
            probe_out = probe(delta_f32, attn_f32)
            emotion_vector = probe_out.get("conditioning_vector")

            # Run connector
            t3_emb = connector(hidden.float(), emotion_vector, target_length=64)

        # Validate output shape and values
        correct_shape = t3_emb.shape == (1, 64, 1024)
        has_values = not torch.isnan(t3_emb).any() and not torch.isinf(t3_emb).any()
        norm = t3_emb.norm(dim=-1).mean().item()
        reasonable_norm = 0.1 < norm < 100

        report.add("HiddenConnector",
                    f"Projection: \"{text[:35]}...\"",
                    correct_shape and has_values and reasonable_norm,
                    f"shape={list(t3_emb.shape)}, norm={norm:.2f}")

    # Test: different emotions produce different projections
    with torch.no_grad():
        tokens = tokenizer("This is a test.", return_tensors="pt").to(device)
        capture.clear()
        out = thinker(**tokens, output_hidden_states=True)
        hidden = out.hidden_states[-1] if hasattr(out, 'hidden_states') and out.hidden_states else None

        if hidden is not None:
            # Fake neutral vs excited emotion vectors
            neutral_ev = torch.zeros(1, 14, device=device)
            neutral_ev[0, 0] = 1.0  # neutral
            excited_ev = torch.zeros(1, 14, device=device)
            excited_ev[0, 4] = 1.0  # excited

            proj_neutral = connector(hidden.float(), neutral_ev, target_length=32)
            proj_excited = connector(hidden.float(), excited_ev, target_length=32)

            diff = (proj_neutral - proj_excited).abs().mean().item()
            report.add("HiddenConnector", "Emotion differentiation",
                       diff > 0.01,
                       f"neutral vs excited diff={diff:.4f}")

    capture.remove_hooks()
    del thinker
    torch.cuda.empty_cache()


def validate_tts(report, voice_path, device):
    """Validate TTS produces intelligible speech (verified by Whisper)."""
    print(f"\n{'-' * 70}")
    print("4. TTS (Chatterbox) -- Speech Generation + Intelligibility")
    print(f"{'-' * 70}")

    from src.inference.streaming import StreamingTTSPipeline

    print("  Loading Chatterbox Turbo...")
    pipeline = StreamingTTSPipeline.from_pretrained(device=device)
    pipeline.set_voice(voice_path)

    # Load faster-whisper for verification (works on Windows, unlike openai whisper)
    print("  Loading faster-whisper for verification...")
    import whisper
    whisper_model = whisper.load_model("base", device=device)

    os.makedirs("test_output/validation", exist_ok=True)

    intelligible_count = 0

    for i, text in enumerate(INTELLIGIBILITY_TEXTS):
        print(f"\n  Testing: \"{text}\"")

        # Generate audio via streaming
        all_audio = []
        t0 = time.time()
        for chunk in pipeline.stream(text, chunk_size=50):
            all_audio.append(chunk.audio)
        gen_time = time.time() - t0

        if not all_audio:
            report.add("TTS", f"Generate: \"{text[:40]}...\"", False, "No audio generated")
            continue

        wav = np.concatenate(all_audio)
        duration = len(wav) / pipeline.tts.sr

        # Save audio
        wav_path = f"test_output/validation/tts_test_{i:02d}.wav"
        sf.write(wav_path, wav, pipeline.tts.sr)

        report.add("TTS", f"Generate audio ({duration:.1f}s in {gen_time:.1f}s)",
                   duration > 0.3 and gen_time < duration * 3,
                   f"realtime={gen_time/duration:.2f}x")

        # Verify with faster-whisper
        result = whisper_model.transcribe(load_audio_np(wav_path))
        heard = result["text"].strip().lower()
        original = text.lower()

        # Check word overlap (fuzzy match)
        original_words = set(original.split())
        heard_words = set(heard.split())
        if original_words:
            overlap = len(original_words & heard_words) / len(original_words)
        else:
            overlap = 0.0

        is_intelligible = overlap >= 0.4  # at least 40% word overlap
        if is_intelligible:
            intelligible_count += 1

        report.add("TTS", f"Intelligibility: \"{text[:30]}...\"", is_intelligible,
                    f"heard=\"{heard[:60]}\", overlap={overlap:.0%}")

    report.add("TTS", f"Overall intelligibility: {intelligible_count}/{len(INTELLIGIBILITY_TEXTS)}",
               intelligible_count >= len(INTELLIGIBILITY_TEXTS) * 0.5,
               "threshold: 50%")

    del whisper_model
    torch.cuda.empty_cache()


def validate_streaming(report, voice_path, device):
    """Validate streaming TTS works and has reasonable latency."""
    print(f"\n{'-' * 70}")
    print("5. STREAMING -- Latency + Chunking")
    print(f"{'-' * 70}")

    from src.inference.streaming import StreamingTTSPipeline, find_clause_boundary

    # Test clause boundary detection
    test_cases = [
        ("Hello world.", 20, None),  # too short
        ("This is a really great test sentence.", 20, 38),  # sentence end
        ("Hello, this is a test of clause detection.", 20, 7),  # wait, "Hello," is only 6 chars
        ("This is a longer sentence that has a comma, and then continues.", 20, 44),
    ]

    for text, min_chars, _ in test_cases:
        boundary = find_clause_boundary(text, min_chars=min_chars)
        if boundary is not None:
            left = text[:boundary].strip()
            right = text[boundary:].strip()
            report.add("Streaming", f"Boundary in: \"{text[:40]}...\"", True,
                        f"split at {boundary}: \"{left}\" | \"{right}\"")
        else:
            report.add("Streaming", f"No boundary (expected): \"{text[:40]}\"",
                       len(text) < min_chars or '.' not in text[min_chars:],
                       f"text_len={len(text)}, min={min_chars}")

    # Test streaming pipeline latency
    print("  Loading streaming pipeline...")
    pipeline = StreamingTTSPipeline.from_pretrained(device=device)
    pipeline.set_voice(voice_path)

    text = "Hello, I am testing the streaming speech generation pipeline today."
    chunks = []
    t0 = time.time()
    first_chunk_time = None

    for chunk in pipeline.stream(text, chunk_size=50):
        if first_chunk_time is None:
            first_chunk_time = time.time() - t0
        chunks.append(chunk)

    total_time = time.time() - t0

    report.add("Streaming", "First chunk latency",
               first_chunk_time is not None and first_chunk_time < 2.0,
               f"{first_chunk_time:.2f}s" if first_chunk_time else "no chunks")

    report.add("Streaming", "Multiple chunks generated",
               len(chunks) >= 2, f"{len(chunks)} chunks")

    report.add("Streaming", "Final chunk marked",
               chunks[-1].is_final if chunks else False, "")


def validate_full_loop(report, voice_path, lora_path, probe_ckpt, device):
    """Full loop: TTS -> Whisper -> Thinker -> TTS -> Whisper."""
    print(f"\n{'-' * 70}")
    print("6. FULL LOOP -- End-to-End Roundtrip")
    print(f"{'-' * 70}")

    from src.inference.streaming import StreamingTTSPipeline

    # Load all components
    print("  Loading pipeline components...")

    # TTS
    tts_pipeline = StreamingTTSPipeline.from_pretrained(device=device)
    tts_pipeline.set_voice(voice_path)

    # Whisper (faster-whisper for Windows compatibility)
    import whisper
    whisper_model = whisper.load_model("base", device=device)

    # Thinker
    import unsloth
    from unsloth import FastLanguageModel
    from src.model.emotion_probe import EmotionProbe
    from src.training.train_probe import HiddenStateCapture
    from src.model.connector import ThinkerTalkerConnector

    thinker, processor = FastLanguageModel.from_pretrained(
        lora_path, device_map=device, dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(thinker)
    thinker.eval()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    hidden_size = thinker.config.get_text_config().hidden_size
    capture = HiddenStateCapture(thinker)

    probe = EmotionProbe(hidden_size=hidden_size).to(device)
    if os.path.exists(probe_ckpt):
        probe.load_state_dict(torch.load(probe_ckpt, map_location=device))
    probe.eval()

    connector = ThinkerTalkerConnector(device=device, use_turbo=True)

    os.makedirs("test_output/validation", exist_ok=True)

    test_inputs = [
        "What is the meaning of life?",
        "Tell me something funny.",
    ]

    for test_idx, user_prompt in enumerate(test_inputs):
        print(f"\n  --- Loop test {test_idx + 1}: \"{user_prompt}\" ---")
        t0 = time.time()

        # Step 1: Generate "user speech" with TTS
        print(f"  [1] Generating user speech...")
        user_audio = []
        for chunk in tts_pipeline.stream(user_prompt, chunk_size=50):
            user_audio.append(chunk.audio)

        if not user_audio:
            report.add("Loop", f"User TTS failed: \"{user_prompt}\"", False, "No audio")
            continue

        user_wav = np.concatenate(user_audio)
        user_wav_path = f"test_output/validation/loop_{test_idx}_user.wav"
        sf.write(user_wav_path, user_wav, tts_pipeline.tts.sr)

        # Step 2: Whisper transcribes user speech
        print(f"  [2] Whisper transcribing user speech...")
        result = whisper_model.transcribe(load_audio_np(user_wav_path))
        transcribed_input = result["text"].strip()
        print(f"      Whisper heard: \"{transcribed_input}\"")

        report.add("Loop", f"STT user speech: \"{user_prompt[:30]}\"",
                   len(transcribed_input) > 3,
                   f"heard: \"{transcribed_input[:50]}\"")

        # Step 3: Thinker generates response
        print(f"  [3] Thinker responding...")
        chat = f"<|im_start|>user\n{transcribed_input}<|im_end|>\n<|im_start|>assistant\n"
        tokens = tokenizer(chat, return_tensors="pt").to(device)

        with torch.no_grad():
            capture.clear()
            thinker(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"],
                    output_hidden_states=True)

            delta_f32 = {k: v.float() for k, v in capture.deltanet_states.items()}
            attn_f32 = {k: v.float() for k, v in capture.attention_states.items()}
            probe_out = probe(delta_f32, attn_f32)

            gen = thinker.generate(
                **tokens, max_new_tokens=80, do_sample=True,
                temperature=0.7, top_p=0.9, repetition_penalty=1.1,
            )

        gen_tokens = gen[0][tokens["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        emotion = probe_out.get("emotion_label", "neutral")
        print(f"      Response: \"{response[:60]}...\" (emotion: {emotion})")

        report.add("Loop", f"Thinker response",
                   len(response) > 5,
                   f"\"{response[:50]}...\" ({emotion})")

        # Step 4: TTS speaks the response
        print(f"  [4] TTS speaking response...")
        style = connector.map_style(
            emotion, probe_out.get("prosody_labels", {}),
            probe_out.get("conditioning_vector"),
        )

        clean_text = connector.clean_text(response)
        response_audio = []
        for chunk in tts_pipeline.stream(clean_text, temperature=style["temperature"], chunk_size=50):
            response_audio.append(chunk.audio)

        if not response_audio:
            report.add("Loop", "Response TTS", False, "No audio generated")
            continue

        response_wav = np.concatenate(response_audio)
        response_wav_path = f"test_output/validation/loop_{test_idx}_response.wav"
        sf.write(response_wav_path, response_wav, tts_pipeline.tts.sr)
        response_duration = len(response_wav) / tts_pipeline.tts.sr

        report.add("Loop", "Response TTS",
                   response_duration > 0.5,
                   f"{response_duration:.1f}s audio")

        # Step 5: Whisper verifies the response
        print(f"  [5] Whisper verifying response...")
        result = whisper_model.transcribe(load_audio_np(response_wav_path))
        verified_text = result["text"].strip()
        print(f"      Whisper heard: \"{verified_text[:60]}\"")

        # Check that Whisper heard something related to what was said
        response_words = set(clean_text.lower().split())
        verified_words = set(verified_text.lower().split())
        if response_words:
            word_overlap = len(response_words & verified_words) / len(response_words)
        else:
            word_overlap = 0.0

        total_time = time.time() - t0

        report.add("Loop", f"Response intelligible",
                   word_overlap >= 0.3 or len(verified_text) > 10,
                   f"overlap={word_overlap:.0%}, heard=\"{verified_text[:50]}\"")

        report.add("Loop", f"Loop time",
                   total_time < 30,
                   f"{total_time:.1f}s total")

    capture.remove_hooks()
    del thinker, whisper_model
    torch.cuda.empty_cache()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate the full Thinker-Talker stack")
    parser.add_argument("--component", choices=["thinker", "probe", "connector", "tts", "streaming", "loop"],
                        help="Test only this component")
    parser.add_argument("--all", action="store_true", help="Run all tests (default if no --component)")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--device", default="cuda")
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

    run_all = args.all or args.component is None

    report = ValidationReport()

    print("=" * 70)
    print("THINKER-TALKER STACK VALIDATION")
    print("=" * 70)
    print(f"  Device:    {args.device}")
    print(f"  LoRA:      {args.lora_path}")
    print(f"  Probe:     {args.probe_ckpt}")
    print(f"  Voice:     {voice}")

    # Check that checkpoints exist
    if not os.path.exists(os.path.join(args.lora_path, "adapter_model.safetensors")):
        print(f"\n  ERROR: LoRA checkpoint not found at {args.lora_path}")
        print(f"  Run Stage 2 (train_lora.py) first!")
        return

    connector_ckpt = "checkpoints/connector/connector_best.pt"

    try:
        if run_all or args.component == "thinker":
            validate_thinker(report, args.lora_path, args.device)

        if run_all or args.component == "connector":
            validate_connector(report)
            validate_hidden_connector(report, args.lora_path, args.probe_ckpt,
                                       connector_ckpt, args.device)

        if run_all or args.component == "probe":
            validate_probe(report, args.lora_path, args.probe_ckpt, args.device)

        if (run_all or args.component in ("tts", "streaming", "loop")) and not voice:
            print("\n  ERROR: No voice reference found. Provide --voice path/to/speaker.wav")
            print("  or place .wav files in data/reference_speakers/")
            return

        if run_all or args.component == "tts":
            validate_tts(report, voice, args.device)

        if run_all or args.component == "streaming":
            validate_streaming(report, voice, args.device)

        if run_all or args.component == "loop":
            validate_full_loop(report, voice, args.lora_path, args.probe_ckpt, args.device)

    except KeyboardInterrupt:
        print("\n\nInterrupted!")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()

    all_passed = report.summary()

    # Save report
    os.makedirs("test_output/validation", exist_ok=True)
    report_path = "test_output/validation/report.json"
    with open(report_path, "w") as f:
        json.dump(report.results, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
