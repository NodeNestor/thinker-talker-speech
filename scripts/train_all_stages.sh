#!/bin/bash
# Train Stage 1, 3, and 4 sequentially.
#
# From Windows:  CUDA_VISIBLE_DEVICES=1 bash scripts/train_all_stages.sh
# From WSL:      cd /mnt/e/Repos/thinker-talker-speech && CUDA_VISIBLE_DEVICES=1 bash scripts/train_all_stages.sh
#
# WSL is recommended — Flash Attention + fla make training 2-3x faster.

set -e
cd "$(dirname "$0")/.."

LORA_PATH="${LORA_PATH:-checkpoints/living-agent/lora}"
CUDA_DEV="${CUDA_VISIBLE_DEVICES:-1}"

# ── WSL venv setup (same as run_wsl.sh) ──
if [ -f /proc/version ] && grep -qi microsoft /proc/version 2>/dev/null; then
    echo "Detected WSL — using .venv-wsl"
    VENV=".venv-wsl"
    if [ ! -d "$VENV" ]; then
        echo "=== First-time WSL setup ==="
        python3 -m venv "$VENV"
        source "$VENV/bin/activate"
        pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
        pip install -q unsloth soundfile numpy librosa peft transformers datasets tqdm bitsandbytes speechbrain torchaudio
        pip install -q flash-attn --no-build-isolation || echo "  flash-attn failed, will use SDPA"
        pip install -q flash-linear-attention || echo "  fla failed, will use O(n^2) fallback"
        echo "=== Setup complete ==="
    else
        source "$VENV/bin/activate"
    fi
fi

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

echo ""
echo "=============================================="
echo "  Stage 3: Emotion Probe"
echo "=============================================="
CUDA_VISIBLE_DEVICES="$CUDA_DEV" python3 -m src.training.train_probe \
    --lora-path "$LORA_PATH" \
    --batch-size 16 \
    --epochs 10 \
    --lr 1e-3 \
    --output checkpoints/probe

echo ""
echo "=============================================="
echo "  Stage 4: Thinker→Talker Connector"
echo "=============================================="
CUDA_VISIBLE_DEVICES="$CUDA_DEV" python3 -m src.training.train_stage4 \
    --lora-path "$LORA_PATH" \
    --probe-ckpt checkpoints/probe/probe_best.pt \
    --batch-size 4 \
    --epochs 8 \
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
