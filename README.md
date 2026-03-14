# Thinker-Talker Speech

End-to-end speech agent: Qwen 3.5 0.8B Thinker + Chatterbox Turbo 350M Talker with emotion-aware streaming TTS.

## Architecture

```
[Mic] -> faster-whisper STT (text for display + history)
      -> WhisperAdapter (audio → Thinker embeddings, preserves tone/prosody)
      -> Qwen 3.5 0.8B Thinker (LoRA, streaming via TextIteratorStreamer)
          |-- tool_call? -> intercept, execute, feed back
          |-- hidden states -> Dual Emotion Probe:
          |     DeltaNet states (18 layers) -> mood/energy/pace
          |     Attention layers (6 layers) -> emphasis/surprise
          |     -> conditioning_vector (14-dim: 10 emotions + 4 prosody)
          |
          |-- PATH A (trained connector):
          |     hidden states -> HiddenStateConnector (~2M) -> T3 text embeddings
          |     Bypasses text decoding — preserves emotion/prosody in embeddings
          |
          |-- PATH B (text fallback):
          |     text (streamed clause-by-clause) -> T3 tokenizer -> T3 text embeddings
          |     Rule-based emotion -> style mapping { temperature, exaggeration }
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
# Full voice mode (mic + TTS) — basic generation
CUDA_VISIBLE_DEVICES=1 python scripts/live_agent.py

# Streaming TTS with reference voice (clause-level, speech-token streaming)
CUDA_VISIBLE_DEVICES=1 python scripts/live_agent.py --voice data/reference_speakers/voice.wav

# Keyboard input, voice output
CUDA_VISIBLE_DEVICES=1 python scripts/live_agent.py --text --voice voice.wav

# Pure text mode
CUDA_VISIBLE_DEVICES=1 python scripts/live_agent.py --text --no-tts
```

Features:
- Always-on mic with VAD (voice activity detection)
- User speech interrupts agent mid-sentence
- **WhisperAdapter**: audio features fed directly to Thinker (preserves tone/prosody beyond text)
- **Emotion Probe**: detects emotion from hidden states, adjusts TTS style
- **Streaming TTS** with `--voice`: clause-level splitting + speech-token streaming (~0.3s first audio)
- **HiddenStateConnector**: when trained, bypasses text decoding for richer speech (auto-detected)
- Tool calling (file ops, git, web search, memory, timers)
- Streaming token display (thinking in gray, speech in white)

## Training Pipeline

### Stage 1: Whisper Adapter

Maps Whisper encoder features into Thinker embedding space. In inference, when mic input is captured, audio features are fed directly to the Thinker via `inputs_embeds` — preserving tone, emphasis, and prosody that text transcription loses. Faster-whisper STT still runs in parallel for text display and conversation history.

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

### Stage 4: HiddenStateConnector

Trains the HiddenStateConnector (~2M params) to project Thinker hidden states directly into T3's text embedding space. This bypasses text decoding entirely — the Thinker's internal representation (which encodes emotion, emphasis, prosody) is projected straight into the TTS model's input space.

Uses FiLM conditioning from the emotion probe + cross-attention for sequence alignment between different tokenizers.

```bash
python -m src.training.train_stage4 --lora-path checkpoints/living-agent/lora --probe-ckpt checkpoints/probe/probe_best.pt
```

**Result**: Loss 5.10 -> **0.090** (cosine similarity ~82%)

When trained, the connector activates automatically in `live_agent.py`. Falls back to text → T3 tokenizer path if checkpoint doesn't exist.

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
| HiddenStateConnector (Thinker→T3) | ~2M | Stage 4 | `checkpoints/connector/` |
| Rule-based style fallback | 0 | No | N/A |
| Chatterbox Turbo Talker | 350M | Frozen | HuggingFace |

## Datasets

| Stage | Dataset | Size | Purpose |
|-------|---------|------|---------|
| 1 | LibriSpeech clean-100 | 100h | Whisper adapter alignment |
| 2 | VoiceAssistant-400K | 400K pairs | Thinker LoRA finetuning |
| 3 | GoEmotions | 58K texts | Emotion probe training |
| 4 | GoEmotions + probe | reused | HiddenStateConnector alignment |

## Demo: AI-to-AI Conversation (End-to-End)

Every audio sample below was generated entirely by AI — no human voice involved. The pipeline generates synthetic "user speech" with TTS, transcribes it with Whisper, feeds it to the Thinker, speaks the response with Chatterbox Turbo, and verifies intelligibility with Whisper. 10/10 validation tests passed.

---

### Conversation 1: "What is the meaning of life?"

**User speaks** (generated by TTS):

<audio controls src="data/demo_samples/loop_0_user.wav"></audio>

> Whisper transcription: *"What is the meaning of life?"* — 100% accurate

**Thinker processes** — reads emotion from hidden states, detects `empathetic` mood:
```
[emotion probe] → confused/empathetic (conditioning_vector: 14-dim)
[hidden states]  → HiddenStateConnector → T3 text embeddings (bypasses text decoding)
```

**Agent responds** (16.2s of streaming speech, 89% intelligibility):

<audio controls src="data/demo_samples/loop_0_response.wav"></audio>

> *"Let me think about this for a bit. Hmm, that's a classic question..."*

---

### Conversation 2: "Tell me something funny."

**User speaks** (generated by TTS):

<audio controls src="data/demo_samples/loop_1_user.wav"></audio>

> Whisper transcription: *"Tell me something funny."* — 100% accurate

**Thinker processes** — thinks internally, then speaks:
```
<thinking>Oh, let me think about something... I can imagine...</thinking>
[streaming TTS] → clause-level splitting → vocode every 50 speech tokens → audio chunks
```

**Agent responds** (21.8s of streaming speech, 76% intelligibility):

<audio controls src="data/demo_samples/loop_1_response.wav"></audio>

> *"Oh let me think about something. I can imagine a funny scenario..."*

---

Full pipeline: Mic/TTS → Whisper STT → WhisperAdapter → Thinker (0.8B) → Emotion Probe → HiddenStateConnector → T3 → Vocode → Speaker. Total VRAM: **~4.8 GB**.

### Full Loop Test

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/validate_stack.py --component loop --voice voice.wav
```

### Overlapped Streaming Benchmark

Compare first-audio latency with and without pipelining:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/test_overlapped.py --compare --voice voice.wav
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/NodeNestor/thinker-talker-speech.git
cd thinker-talker-speech
pip install -r requirements.txt

# 2. Download pre-trained components
python scripts/download_models.py

# 3. Download training data
python scripts/download_data.py

# 4. Train (picks GPU automatically)
LORA_PATH=checkpoints/living-agent/lora bash scripts/train_all_stages.sh

# 5. Validate everything works
CUDA_VISIBLE_DEVICES=1 python scripts/validate_stack.py --voice data/reference_speakers/voice.wav

# 6. Run the live agent
CUDA_VISIBLE_DEVICES=1 python scripts/live_agent.py --voice data/reference_speakers/voice.wav
```

## Requirements

- NVIDIA GPU with >= 6GB VRAM (~4.8 GB used at runtime)
- Python 3.10+
- PyTorch 2.x with CUDA
- See `requirements.txt` for full dependency list

### Platform Notes

- **Windows**: `torchaudio.save` requires TorchCodec — pipeline uses `soundfile` instead
- **WSL**: Recommended for training — supports Flash Attention (2-3x faster)
- **CUDA devices**: `CUDA_VISIBLE_DEVICES=0` = RTX 4060 (8GB), `CUDA_VISIBLE_DEVICES=1` = RTX 5060 Ti (16GB)
