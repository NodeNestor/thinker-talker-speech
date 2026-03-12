# Thinker-Talker Speech

End-to-end speech model: Qwen 3.5 0.8B Thinker + pre-trained Talker head with emotion/prosody probing from DeltaNet states.

## Architecture

```
[Mic] -> Whisper-small (244M, frozen)
      -> CNN Adapter (downsample 50Hz->25Hz, project 768->1024)
      -> Qwen 3.5 0.8B Thinker (LoRA)
          |-- tool_call? -> intercept, execute, feed back
          |-- hidden states -> Dual Emotion Probe:
          |     DeltaNet states -> mood/energy/pace (slow-moving context)
          |     Attention layers -> emphasis/surprise (per-token)
          |     -> [emotion_vec, prosody_vec]
          |-- text tokens + hidden states + emotion/prosody
                  |
[Voice sample] -> ECAPA-TDNN -> speaker embedding (192-dim)
                  |                    |
              Chatterbox Turbo (350M) Talker
              (conditioned on: thinker hidden states
                             + emotion/prosody vectors via AdaLN
                             + speaker embedding via AdaLN)
                  |
              Audio stream -> Speaker
```

## Key Design Decisions

- **Thinker stays in text space** -- tool calling works naturally
- **Emotion is READ, not generated** -- SAE/probe extracts emotion from DeltaNet recurrent states
- **Voice is a runtime parameter** -- 3s sample -> ECAPA embedding -> swap anytime
- **Prosody is continuous** -- speed, pitch, energy as float vectors, not discrete tags

## Components

| Component | Size | Training Required? |
|-----------|------|-------------------|
| Whisper-small encoder | 244M | No (frozen) |
| CNN Adapter | ~2M | Yes (Stage 1) |
| Qwen 3.5 0.8B Thinker | 0.8B | LoRA only (~16M) |
| Emotion/Prosody Probe | ~1-5M | Yes (labeled emotion data) |
| Chatterbox Turbo Talker | 350M | No (pre-trained) |
| ECAPA-TDNN speaker encoder | ~7M | No (pre-trained) |
| Thinker->Talker connector | ~5M | Yes (Stage 3) |

## Training Stages

1. **Adapter pretraining** -- LibriSpeech, teach adapter to map Whisper features to Thinker embedding space
2. **Thinker LoRA** -- VoiceAssistant-400K, speech instruction following
3. **Emotion probe** -- GoEmotions + MELD + UltraVoice, extract emotion/prosody from hidden states
4. **Connector** -- Wire Thinker hidden states to Chatterbox Talker

## Datasets

| Stage | Dataset | Size | HuggingFace ID |
|-------|---------|------|----------------|
| 1 | LibriSpeech clean-360 | 360h | `openslr/librispeech_asr` |
| 2 | VoiceAssistant-400K | 400K pairs | `gpt-omni/VoiceAssistant-400K` |
| 3 | GoEmotions | 58K texts, 28 emotions | `google-research-datasets/go_emotions` |
| 3 | UltraVoice | 833h, emotion+speed+volume | `tutu0604/UltraVoice` |
| 3 | Expresso | ~40h, 7 styles | `ylacombe/expresso` |
| 3 | MELD | 13K utts, 7 emotions | `ajyy/MELD_audio` |
| 4 | LibriTTS-R | 585h, 2456 speakers | `mythicinfinity/libritts_r` |

## Quick Start

```bash
pip install -r requirements.txt
python scripts/download_models.py    # Download pre-trained components
python scripts/download_data.py      # Download training datasets
python scripts/train_stage1.py       # Train adapter
python scripts/train_stage2.py       # LoRA tune Thinker
python scripts/train_stage3.py       # Train emotion probe
python scripts/train_stage4.py       # Train connector
python src/inference/demo.py         # Run end-to-end demo
```
