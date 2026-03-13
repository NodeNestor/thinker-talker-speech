# Thinker-Talker Speech

End-to-end speech agent: Qwen 3.5 0.8B Thinker + Chatterbox Turbo 350M Talker with emotion-aware streaming TTS.

## Architecture

```
[Mic] -> faster-whisper STT (real-time transcription)
      -> Text input
      -> Qwen 3.5 0.8B Thinker (LoRA, streaming via TextIteratorStreamer)
          |-- tool_call? -> intercept, execute, feed back
          |-- hidden states -> Dual Emotion Probe:
          |     DeltaNet states (18 layers) -> mood/energy/pace
          |     Attention layers (6 layers) -> emphasis/surprise
          |     -> conditioning_vector (14-dim: 10 emotions + 4 prosody)
          |-- text (streamed clause-by-clause)
                  |
          Connector (rule-based emotion -> style mapping)
          emotion_label -> { exaggeration, cfg_weight, temperature }
                  |
          Chatterbox Turbo (350M) — streaming TTS
          T3 generates speech tokens -> vocode every 50 tokens -> audio chunks
                  |
          Audio stream -> Speaker (real-time playback via sounddevice)
```

### Streaming Pipeline

The system uses **overlapped streaming** — the Thinker generates text while the TTS generates audio:

```
Time    Thinker                          TTS
0.0s    Start generating text            (waiting)
0.1s    "I'm really excited!" ready ->   Start T3 on clause 1
0.3s    Still generating text...         First audio chunk playing!
0.5s    Text generation done             Still speaking clause 1...
0.8s    (idle)                           Start clause 2
```

First audio in ~0.3s instead of ~1.5s without overlap.

## Live Agent

The live agent runs with concurrent mic listening, thinking, and speaking:

```bash
# Full voice mode (mic + TTS)
CUDA_VISIBLE_DEVICES=1 python scripts/live_agent.py

# Keyboard input, voice output
CUDA_VISIBLE_DEVICES=1 python scripts/live_agent.py --text

# Pure text mode
CUDA_VISIBLE_DEVICES=1 python scripts/live_agent.py --text --no-tts
```

Features:
- Always-on mic with VAD (voice activity detection)
- User speech interrupts agent mid-sentence
- Emotion-aware TTS (probe detects emotion, connector maps to Chatterbox style)
- Tool calling (file ops, git, web search, memory, timers)
- Streaming token display (thinking in gray, speech in white)

## Training Pipeline

### Stage 1: Whisper Adapter

Maps Whisper encoder features into Thinker embedding space (for future end-to-end speech input).

```bash
python -m src.training.train_stage1 --lora-path checkpoints/living-agent/lora
```

**Result**: Loss 0.52 -> **0.17** (cosine similarity ~0.83)

### Stage 2: Thinker LoRA

Standard LoRA finetuning on VoiceAssistant-400K for speech instruction following.

### Stage 3: Emotion Probe

Trains a ~2M param probe to read emotion and prosody from the Thinker's frozen hidden states.

**Architecture**: `HiddenStateCapture` registers forward hooks on all 24 Qwen 3.5 layers. DeltaNet layers (18) feed a `DeltaNetProbe` and attention layers (6) feed an `AttentionProbe`. Both produce a 14-dim conditioning vector (10 emotions + 4 prosody floats).

**Similarity-weighted soft labels**: Confusing similar emotions costs less than dissimilar ones. Groups: positive (happy/excited/calm), negative (sad/fearful/disgusted), hostile (angry), reactive (surprised/confused).

```bash
python -m src.training.train_probe --lora-path checkpoints/living-agent/lora --epochs 15
```

**Result**: 50.4% accuracy (10-class, soft labels)

### Stage 4: StyleMapper (Optional)

Trains a small network (~10K params) to learn emotion→style mapping from the probe's conditioning vector. Falls back to rule-based lookup tables if not trained.

The connector is **rule-based by default** — it maps emotion labels to Chatterbox params via lookup tables (e.g., "excited" → exaggeration=0.9, temperature=0.95). The StyleMapper learns to generalize beyond discrete labels.

```bash
python -m src.training.train_stage4 --lora-path checkpoints/living-agent/lora --probe-ckpt checkpoints/probe/probe_best.pt
```

## Validation

Comprehensive stack validation — tests every component individually and end-to-end:

```bash
# Full validation (all components)
CUDA_VISIBLE_DEVICES=1 python scripts/validate_stack.py

# Individual components
CUDA_VISIBLE_DEVICES=1 python scripts/validate_stack.py --component thinker
CUDA_VISIBLE_DEVICES=1 python scripts/validate_stack.py --component probe
CUDA_VISIBLE_DEVICES=1 python scripts/validate_stack.py --component tts
CUDA_VISIBLE_DEVICES=1 python scripts/validate_stack.py --component loop  # full TTS→STT roundtrip
```

The loop test validates end-to-end: generates "user speech" with TTS → transcribes with Whisper → feeds to Thinker → speaks response → verifies with Whisper.

## Streaming Test

Compare overlapped vs non-overlapped streaming:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/test_overlapped.py --compare
```

## Run All Stages

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/train_all_stages.sh
```

## Components

| Component | Size | Training? | Checkpoint |
|-----------|------|-----------|------------|
| Whisper-small encoder | 244M | Frozen | HuggingFace |
| CNN Adapter (Whisper→Thinker) | ~2M | Stage 1 | `checkpoints/adapter/` |
| Qwen 3.5 0.8B Thinker | 0.8B | LoRA (~16M) | `checkpoints/living-agent/lora/` |
| Emotion/Prosody Probe | ~2M | Stage 3 | `checkpoints/probe/` |
| Connector (rule-based) | 0 | No | N/A |
| StyleMapper (optional) | ~10K | Stage 4 | `checkpoints/connector/` |
| ECAPA-TDNN speaker encoder | ~7M | Frozen | SpeechBrain |
| Chatterbox Turbo Talker | 350M | Frozen | HuggingFace |

## Datasets

| Stage | Dataset | Size | Purpose |
|-------|---------|------|---------|
| 1 | LibriSpeech clean-100 | 100h | Whisper adapter alignment |
| 2 | VoiceAssistant-400K | 400K pairs | Thinker LoRA finetuning |
| 3 | GoEmotions | 58K texts | Emotion probe training |
| 4 | GoEmotions + probe | reused | StyleMapper targets |

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/user/thinker-talker-speech.git
cd thinker-talker-speech
pip install -r requirements.txt

# 2. Download pre-trained components
python scripts/download_models.py

# 3. Download training data
python scripts/download_data.py

# 4. Train (picks GPU automatically)
LORA_PATH=checkpoints/living-agent/lora bash scripts/train_all_stages.sh

# 5. Validate everything works
CUDA_VISIBLE_DEVICES=1 python scripts/validate_stack.py

# 6. Run the live agent
CUDA_VISIBLE_DEVICES=1 python scripts/live_agent.py
```

## Requirements

- NVIDIA GPU with >= 8GB VRAM (16GB recommended)
- Python 3.10+
- PyTorch 2.x with CUDA
- See `requirements.txt` for full dependency list

### Platform Notes

- **Windows**: `torchaudio.save` requires TorchCodec — pipeline uses `soundfile` instead
- **WSL**: Recommended for training — supports Flash Attention (2-3x faster)
- **CUDA devices**: `CUDA_VISIBLE_DEVICES=0` = RTX 4060 (8GB), `CUDA_VISIBLE_DEVICES=1` = RTX 5060 Ti (16GB)
