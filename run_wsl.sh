#!/bin/bash
# Run the living agent in WSL for maximum speed (Flash Attention + fla)
# Usage: just run ./run_wsl.sh from WSL, or `wsl bash run_wsl.sh` from PowerShell

set -e
cd "$(dirname "$0")"

VENV=".venv-wsl"
CUDA_DEV="${CUDA_VISIBLE_DEVICES:-1}"

# ── First-time setup ───────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "=== First-time setup (this takes a few minutes) ==="

    # Check CUDA
    if ! command -v nvidia-smi &>/dev/null; then
        echo "ERROR: nvidia-smi not found. Make sure you have WSL2 + NVIDIA GPU drivers."
        exit 1
    fi
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

    python3 -m venv "$VENV"
    source "$VENV/bin/activate"

    echo "Installing PyTorch..."
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124

    echo "Installing core deps..."
    pip install -q unsloth faster-whisper sounddevice soundfile numpy librosa peft transformers

    echo "Installing Flash Attention (this compiles, takes a while)..."
    pip install -q flash-attn --no-build-isolation || echo "  flash-attn failed, will use SDPA fallback"

    echo "Installing flash-linear-attention (DeltaNet fast path)..."
    pip install -q flash-linear-attention || echo "  fla failed, will use O(n^2) fallback"

    echo "Installing Chatterbox TTS..."
    pip install -q chatterbox-tts --no-deps
    pip install -q resemble-perth s3tokenizer conformer pyloudnorm "numpy<2.4"

    echo "=== Setup complete ==="
else
    source "$VENV/bin/activate"
fi

# ── Run ────────────────────────────────────────────────────────────
echo ""
CUDA_VISIBLE_DEVICES="$CUDA_DEV" python3 scripts/live_agent.py "$@"
