#!/bin/bash
# Train all stages sequentially: Adapter → Probe → Synthetic Data → Connector
#
# Usage:
#   bash scripts/train_all_stages.sh                          # uses defaults
#   CUDA_VISIBLE_DEVICES=1 bash scripts/train_all_stages.sh   # pick GPU
#   LORA_PATH=path/to/lora bash scripts/train_all_stages.sh   # custom LoRA
#
# From WSL (recommended — Flash Attention + fla make training 2-3x faster):
#   cd /mnt/e/Repos/thinker-talker-speech
#   CUDA_VISIBLE_DEVICES=1 bash scripts/train_all_stages.sh
#
# Prerequisites:
#   pip install -r requirements.txt
#   python scripts/download_models.py   # downloads Whisper, ECAPA-TDNN, etc.
#   You need a LoRA checkpoint at $LORA_PATH (default: checkpoints/living-agent/lora)

set -e
cd "$(dirname "$0")/.."

LORA_PATH="${LORA_PATH:-checkpoints/living-agent/lora}"
CUDA_DEV="${CUDA_VISIBLE_DEVICES:-0}"
SYNTHETIC_DIR="data/synthetic_connector"
NUM_PER_EMOTION="${NUM_PER_EMOTION:-75}"

# ── Validate prerequisites ──
if [ ! -d "$LORA_PATH" ]; then
    echo "ERROR: LoRA checkpoint not found at $LORA_PATH"
    echo "Set LORA_PATH to your Thinker LoRA checkpoint directory."
    exit 1
fi

# ── WSL venv setup ──
if [ -f /proc/version ] && grep -qi microsoft /proc/version 2>/dev/null; then
    echo "Detected WSL — using .venv-wsl"
    VENV=".venv-wsl"
    if [ ! -d "$VENV" ]; then
        echo "=== First-time WSL setup ==="
        python3 -m venv "$VENV"
        source "$VENV/bin/activate"
        pip install -q -r requirements.txt
        pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
        pip install -q flash-attn --no-build-isolation || echo "  flash-attn failed, will use SDPA"
        pip install -q flash-linear-attention || echo "  fla failed, will use O(n^2) fallback"
        echo "=== Setup complete ==="
    else
        source "$VENV/bin/activate"
    fi
fi

echo "=============================================="
echo "  Thinker-Talker Training Pipeline"
echo "=============================================="
echo "  LoRA:  $LORA_PATH"
echo "  GPU:   CUDA_VISIBLE_DEVICES=$CUDA_DEV"
echo ""

# ── Stage 1: Whisper Adapter ──
echo "=============================================="
echo "  Stage 1: Whisper Adapter"
echo "=============================================="
CUDA_VISIBLE_DEVICES="$CUDA_DEV" python3 -m src.training.train_stage1 \
    --lora-path "$LORA_PATH" \
    --whisper "openai/whisper-small" \
    --batch-size 8 \
    --epochs 5 \
    --lr 3e-4 \
    --grad-accum 4 \
    --output checkpoints/adapter

# ── Stage 3: Emotion Probe ──
echo ""
echo "=============================================="
echo "  Stage 3: Emotion Probe"
echo "=============================================="
CUDA_VISIBLE_DEVICES="$CUDA_DEV" python3 -m src.training.train_probe \
    --lora-path "$LORA_PATH" \
    --batch-size 16 \
    --epochs 15 \
    --lr 1e-3 \
    --output checkpoints/probe

# ── Synthetic Data Generation ──
echo ""
echo "=============================================="
echo "  Generating Synthetic Training Data"
echo "=============================================="
echo "  ${NUM_PER_EMOTION} samples per emotion x 8 emotions = $((NUM_PER_EMOTION * 8)) total"
CUDA_VISIBLE_DEVICES="$CUDA_DEV" python3 -m src.training.generate_connector_data \
    --lora-path "$LORA_PATH" \
    --output "$SYNTHETIC_DIR" \
    --num-per-emotion "$NUM_PER_EMOTION" \
    --device cuda

# ── Stage 4: Thinker→Talker Connector ──
echo ""
echo "=============================================="
echo "  Stage 4: Thinker→Talker Connector"
echo "=============================================="
CUDA_VISIBLE_DEVICES="$CUDA_DEV" python3 -m src.training.train_stage4 \
    --lora-path "$LORA_PATH" \
    --probe-ckpt checkpoints/probe/probe_best.pt \
    --synthetic-data "$SYNTHETIC_DIR/manifest.json" \
    --batch-size 4 \
    --epochs 12 \
    --lr 3e-4 \
    --grad-accum 4 \
    --output checkpoints/connector

echo ""
echo "=============================================="
echo "  All stages complete!"
echo "=============================================="
echo "Checkpoints:"
echo "  Adapter:   checkpoints/adapter/"
echo "  Probe:     checkpoints/probe/"
echo "  Connector: checkpoints/connector/"
echo ""
echo "Synthetic data: $SYNTHETIC_DIR/"
