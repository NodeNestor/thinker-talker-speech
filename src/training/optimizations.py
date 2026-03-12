"""Training optimizations — CCE, Apollo Mini, Liger kernels, etc.

All memory/compute optimizations in one place so training scripts stay clean.
"""

import torch
import torch.nn as nn
from typing import Optional


# =============================================================================
# Cut Cross-Entropy (Apple, ICLR 2025)
# Computes CE loss WITHOUT materializing the full logit matrix.
# For Qwen 3.5 with vocab=248,320 this saves gigabytes of memory.
# =============================================================================

def get_cce_loss_fn():
    """Get the CCE loss function. Falls back to standard CE if unavailable."""
    try:
        from cut_cross_entropy import linear_cross_entropy
        print("[opt] Cut Cross-Entropy loaded — logit matrix will NOT be materialized")
        return linear_cross_entropy
    except ImportError:
        print("[opt] cut-cross-entropy not installed, using standard CE")
        return None


def compute_loss_cce(hidden_states, classifier_weight, labels, shift=True):
    """Compute language modeling loss using CCE.

    Args:
        hidden_states: [batch, seq, hidden] from last layer
        classifier_weight: [vocab, hidden] the lm_head weight
        labels: [batch, seq] token IDs, -100 for ignored positions
        shift: Whether to shift for causal LM (default True)
    """
    cce_fn = get_cce_loss_fn()
    if cce_fn is not None:
        return cce_fn(hidden_states, classifier_weight, labels, shift=1 if shift else 0)
    else:
        # Fallback: standard cross entropy
        if shift:
            hidden_states = hidden_states[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
        logits = hidden_states @ classifier_weight.T
        return nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )


# =============================================================================
# Apollo Mini Optimizer (MLSys 2025)
# SGD-level memory cost with AdamW-level performance.
# Rank 1 + per-tensor scaling = ~1/1024 of AdamW optimizer states.
# =============================================================================

def get_apollo_mini_optimizer(model, lr=1e-4, weight_decay=0.01):
    """Create Apollo Mini optimizer with proper param groups.

    Separates parameters into:
    - Low-rank params (attention, MLP) -> Apollo Mini with rank 1
    - Non-low-rank params (embeddings, norms) -> Standard AdamW
    """
    try:
        from apollo_torch import APOLLOAdamW
        print("[opt] Apollo Mini optimizer loaded — ~1/1024 memory vs AdamW")
    except ImportError:
        print("[opt] apollo-torch not installed, falling back to AdamW 8-bit")
        return get_adamw_8bit_optimizer(model, lr, weight_decay)

    lowrank_params = []
    standard_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Low-rank for projection matrices, standard for everything else
        if any(m in name for m in ["proj", "gate", "up_proj", "down_proj", "lora_"]):
            lowrank_params.append(param)
        else:
            standard_params.append(param)

    param_groups = [
        {"params": standard_params, "lr": lr, "weight_decay": weight_decay},
        {
            "params": lowrank_params,
            "lr": lr,
            "weight_decay": weight_decay,
            "rank": 1,
            "proj": "random",
            "scale_type": "tensor",
            "scale": 128,
            "update_proj_gap": 200,
            "proj_type": "std",
        },
    ]

    return APOLLOAdamW(param_groups, lr=lr, weight_decay=weight_decay)


def get_adamw_8bit_optimizer(model, lr=1e-4, weight_decay=0.01):
    """Fallback: 8-bit AdamW from bitsandbytes."""
    try:
        import bitsandbytes as bnb
        print("[opt] AdamW 8-bit loaded — 2x memory savings vs fp32 AdamW")
        return bnb.optim.AdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
    except ImportError:
        print("[opt] bitsandbytes not installed, using standard AdamW")
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )


# =============================================================================
# Liger Kernel (LinkedIn) — fused Triton kernels
# +20% throughput, -60% memory for RMSNorm, SwiGLU, RoPE, CrossEntropy
# =============================================================================

def apply_liger_kernels():
    """Apply Liger Kernel optimizations globally."""
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2
        # Try Qwen 3.5 specific kernels first
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
            apply_liger_kernel_to_qwen3_5()
            print("[opt] Liger kernels applied for Qwen 3.5")
            return True
        except (ImportError, AttributeError):
            # Fall back to Qwen2 kernels (RMSNorm, SwiGLU are the same)
            apply_liger_kernel_to_qwen2()
            print("[opt] Liger kernels applied (Qwen2 variant — RMSNorm + SwiGLU)")
            return True
    except ImportError:
        print("[opt] liger-kernel not installed, skipping")
        return False


# =============================================================================
# Combined setup
# =============================================================================

def setup_training_optimizations(model, lr=1e-4, weight_decay=0.01):
    """Apply all optimizations and return optimizer.

    Call this once before training starts.
    Returns: (optimizer, cce_loss_fn_or_None)
    """
    print("=" * 50)
    print("Setting up training optimizations")
    print("=" * 50)

    # 1. Liger kernels (must be before model forward)
    apply_liger_kernels()

    # 2. Gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[opt] Gradient checkpointing enabled")

    # 3. torch.compile (optional — may conflict with some custom modules)
    # Uncomment if needed:
    # model = torch.compile(model, mode="reduce-overhead")
    # print("[opt] torch.compile applied")

    # 4. Optimizer
    optimizer = get_apollo_mini_optimizer(model, lr=lr, weight_decay=weight_decay)

    # 5. CCE loss
    cce_fn = get_cce_loss_fn()

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[params] Trainable: {trainable:,} ({trainable/total*100:.1f}%)")
    print(f"[params] Total: {total:,}")
    print(f"[params] Frozen: {total - trainable:,}")
    print("=" * 50)

    return optimizer, cce_fn
