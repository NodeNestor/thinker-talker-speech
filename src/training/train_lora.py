"""
Fine-tune Qwen3.5 on living agent data using Unsloth QLoRA.

Ported from ClaudeCodeFinetune/train.py, adapted for the living agent format:
  - Loads JSONL with {"turns": [{"role": ..., "content": ...}]} format
  - Masks <tool_result> blocks WITHIN assistant turns (environment responses, not model output)
  - Keeps all ClaudeCodeFinetune optimizations: CCE, fp8, int8 embed, SDPA, RMSNorm bf16

Supports all Qwen3.5 sizes — they share the same hybrid DeltaNet/GQA architecture:
  Dense:  0.8B, 2B, 4B, 9B, 27B
  MoE:    35B-A3B, 122B-A10B, 397B-A17B

Usage:
    python train_lora.py                                     # defaults (0.8B, 16K context)
    python train_lora.py --model unsloth/Qwen3.5-0.8B        # explicit tiny
    python train_lora.py --model unsloth/Qwen3.5-9B           # bigger
    python train_lora.py --seq-length 32768                   # longer context
    python train_lora.py --epochs 2 --lr 1e-4                 # custom hyperparams
"""

import argparse
import os
import sys
import subprocess

# Auto-install wandb if not present (avoids Docker rebuild)
try:
    import wandb
except ImportError:
    if os.environ.get("WANDB_API_KEY"):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", "wandb"])
            import wandb
        except Exception:
            print("  wandb install failed, continuing without logging")

# ─── Patch CCE + Unsloth for native fp8 lm_head BEFORE any imports ───
# Must happen before import unsloth/cut_cross_entropy so Triton JIT compiles patched source.
def _patch_cce_for_fp8():
    """Patch 4 lines in CCE/Unsloth so Triton kernels handle fp8 natively (zero copy)."""
    import importlib
    import importlib.util
    _cce_spec = importlib.util.find_spec('cut_cross_entropy')
    _zoo_spec = importlib.util.find_spec('unsloth_zoo')
    if not _cce_spec or not _zoo_spec:
        return
    _cce_pkg = os.path.dirname(_cce_spec.origin)
    _zoo_pkg = os.path.dirname(_zoo_spec.origin)

    patched = []

    # 1. indexed_dot.py: add .to(e.dtype) on tl.load(c_ptrs)
    p = os.path.join(_cce_pkg, 'indexed_dot.py')
    with open(p) as f:
        src = f.read()
    if 'tl.load(c_ptrs)' in src and '.to(e.dtype)' not in src.split('def indexed_neg_dot')[0]:
        src = src.replace(
            'c = tl.load(c_ptrs)\n',
            'c = tl.load(c_ptrs).to(e.dtype)\n'
        ).replace(
            'c = tl.load(c_ptrs, mask=offs_d[None, :] < D, other=0.0)\n',
            'c = tl.load(c_ptrs, mask=offs_d[None, :] < D, other=0.0).to(e.dtype)\n'
        )
        with open(p, 'w') as f:
            f.write(src)
        patched.append('indexed_dot')

    # 2. cce_backward.py: remove bf16/fp16/fp32 assert
    p = os.path.join(_cce_pkg, 'cce_backward.py')
    with open(p) as f:
        src = f.read()
    if '"Backwards requires classifier' in src:
        src = src.replace(
            'assert c.dtype in (\n        torch.float16,\n        torch.bfloat16,\n        torch.float32,\n    ), "Backwards requires classifier to be bf16 or fp16 or fp32"',
            'pass  # fp8 OK: Triton kernel casts via .to(e.dtype)'
        )
        with open(p, 'w') as f:
            f.write(src)
        patched.append('cce_backward')

    # 3. loss_utils.py: don't cast hidden_states to lm_weight.dtype
    p = os.path.join(_zoo_pkg, 'loss_utils.py')
    with open(p) as f:
        src = f.read()
    if 'hidden_states.to(lm_weight.dtype)' in src:
        src = src.replace(
            'hidden_states.to(lm_weight.dtype)',
            'hidden_states'
        )
        with open(p, 'w') as f:
            f.write(src)
        patched.append('loss_utils')

    # Clear __pycache__ for patched files
    for pkg_dir in [_cce_pkg, _zoo_pkg]:
        cache = os.path.join(pkg_dir, '__pycache__')
        if os.path.isdir(cache):
            import shutil
            shutil.rmtree(cache, ignore_errors=True)

    if patched:
        print(f"  fp8 patches applied: {', '.join(patched)}")

_patch_cce_for_fp8()

import unsloth  # Must be imported before transformers

# CRITICAL: Unsloth resets UNSLOTH_ENABLE_CCE=0 during import.
# Set it back to "1" AFTER import — the compiled module reads it lazily at first forward pass.
os.environ["UNSLOTH_ENABLE_CCE"] = "1"
import gc
import re
import json
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel


# =============================================================================
# Dataset format conversion — living agent JSONL → ChatML text with label masks
# =============================================================================

TOOL_RESULT_PATTERN = re.compile(r"<tool_result>.*?</tool_result>", re.DOTALL)


def format_living_agent_to_chatml(example, tokenizer):
    """Convert living agent format to ChatML text via tokenizer.

    Input: {"turns": [{"role": "user/assistant/system", "content": "..."}]}
    Output: {"text": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n..."}
    """
    turns = example.get("turns", [])
    messages = []
    has_user = False

    for turn in turns:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role not in ("user", "assistant", "system"):
            continue
        if role == "user":
            has_user = True
        messages.append({"role": role, "content": content})

    if not has_user or len(messages) < 2:
        return {"text": ""}

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return {"text": ""}
    return {"text": text}


def build_tool_result_mask(text: str, tokenizer) -> list[bool]:
    """Build a per-token mask that's True for tokens we SHOULD train on.

    Masks out:
      - All user/system turns (handled by SFTTrainer natively)
      - <tool_result>...</tool_result> blocks within assistant turns (our custom addition)

    Returns a list of bools, one per token. True = train on this token.
    """
    # Tokenize the full text
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    n_tokens = len(token_ids)

    # Start with all True (SFTTrainer already masks non-assistant tokens)
    mask = [True] * n_tokens

    # Find all <tool_result>...</tool_result> spans in the text
    for m in TOOL_RESULT_PATTERN.finditer(text):
        start_char, end_char = m.start(), m.end()

        # Find which tokens correspond to this char range
        # We need to map char positions to token positions
        # Decode token by token to find the mapping
        char_pos = 0
        for tok_idx in range(n_tokens):
            tok_text = tokenizer.decode([token_ids[tok_idx]])
            tok_start = char_pos
            tok_end = char_pos + len(tok_text)

            # If this token overlaps with the tool_result span, mask it
            if tok_start < end_char and tok_end > start_char:
                mask[tok_idx] = False

            char_pos = tok_end

    return mask


class LivingAgentDataCollator:
    """Custom data collator that masks <tool_result> blocks in addition to
    the standard user/system masking that SFTTrainer does.

    This ensures we only compute loss on model-generated tokens:
      - <think> blocks ✓ train
      - <tool_call> blocks ✓ train
      - <speak> blocks ✓ train
      - <interrupted/> ✓ train
      - <tool_result> blocks ✗ mask (environment response)
      - User content ✗ mask (handled by SFTTrainer)
      - System content ✗ mask (handled by SFTTrainer)
    """

    def __init__(self, tokenizer, tool_result_open="<tool_result>", tool_result_close="</tool_result>"):
        self.tokenizer = tokenizer
        # Pre-tokenize the markers so we can find them in token space
        self.open_ids = tokenizer.encode(tool_result_open, add_special_tokens=False)
        self.close_ids = tokenizer.encode(tool_result_close, add_special_tokens=False)

    def mask_tool_results_in_labels(self, labels: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Set labels to -100 for all tokens inside <tool_result>...</tool_result> spans.

        Works in token space — finds the open/close token sequences and masks everything between.
        """
        # Work on each sequence in the batch
        for b in range(input_ids.shape[0]):
            ids = input_ids[b].tolist()
            lab = labels[b]

            # Find all <tool_result> open positions
            open_len = len(self.open_ids)
            close_len = len(self.close_ids)
            in_result = False
            i = 0
            while i < len(ids):
                if not in_result:
                    # Check for open tag
                    if ids[i:i + open_len] == self.open_ids:
                        # Mask the open tag and everything after until close
                        for j in range(i, min(i + open_len, len(ids))):
                            lab[j] = -100
                        in_result = True
                        i += open_len
                        continue
                else:
                    # Mask current token
                    lab[i] = -100
                    # Check for close tag
                    if ids[i:i + close_len] == self.close_ids:
                        # Mask the close tag tokens too
                        for j in range(i, min(i + close_len, len(ids))):
                            lab[j] = -100
                        in_result = False
                        i += close_len
                        continue
                i += 1

        return labels


# =============================================================================
# Config loading (same pattern as ClaudeCodeFinetune)
# =============================================================================

def load_config_file(path="config.yaml"):
    """Load config from YAML file if it exists. Returns dict of overrides."""
    if not os.path.isfile(path):
        return {}
    try:
        import yaml
    except ImportError:
        config = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if val.startswith("#"):
                    continue
                if " #" in val:
                    val = val[:val.index(" #")].strip()
                if val.lower() in ("true", "yes"):
                    config[key] = True
                elif val.lower() in ("false", "no"):
                    config[key] = False
                elif val.replace(".", "", 1).replace("-", "", 1).replace("e", "", 1).isdigit():
                    config[key] = float(val) if "." in val or "e" in val.lower() else int(val)
                else:
                    config[key] = val
        return config
    else:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3.5 on living agent data (LoRA + CCE + fp8)",
        epilog="Config file: put a config.yaml in the working directory. CLI flags override config values.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--model", default=None, help="Base model (default: unsloth/Qwen3.5-0.8B)")
    parser.add_argument("--dataset", default=None, help="Dataset path (JSONL)")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--seq-length", type=int, default=None, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=None, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha")
    parser.add_argument("--eval-split", type=float, default=None, help="Eval split ratio")
    parser.add_argument("--save-steps", type=int, default=None, help="Save checkpoint every N steps")
    parser.add_argument("--packing", action="store_true", default=None, help="Enable sequence packing")
    parser.add_argument("--export-gguf", action="store_true", default=None, help="Export GGUF after training")
    parser.add_argument("--export-merged", action="store_true", default=None, help="Export merged model")
    parser.add_argument("--profile-tensors", action="store_true", default=None, help="Profile saved tensors on step 0")
    parser.add_argument("--profile-deep", action="store_true", default=None, help="Deep memory profile")
    parser.add_argument("--offload-layers", type=int, default=None, help="Layers to offload to GPU 1")
    parser.add_argument("--warmup-ratio", type=float, default=None, help="Warmup ratio")
    parser.add_argument("--mask-tool-results", action="store_true", default=None,
                        help="Mask <tool_result> blocks in loss (default: True)")
    parser.add_argument("--quant-embed", action="store_true", default=None,
                        help="Quantize embed_tokens to int8 (saves ~1GB, good for 9B+)")
    parser.add_argument("--no-quant-embed", action="store_true", default=None,
                        help="Disable embed_tokens int8 quantization")
    parser.add_argument("--quant-lmhead-fp8", action="store_true", default=None,
                        help="Quantize lm_head to fp8 after init (saves ~1GB, good for 9B+)")
    parser.add_argument("--no-quant-lmhead-fp8", action="store_true", default=None,
                        help="Disable lm_head fp8 quantization")
    parser.add_argument("--patch-rmsnorm-bf16", action="store_true", default=None,
                        help="Patch RMSNorm to bf16 (saves ~40%% activation memory)")
    parser.add_argument("--no-patch-rmsnorm-bf16", action="store_true", default=None,
                        help="Disable RMSNorm bf16 patch")
    cli_args = parser.parse_args()

    # Layer: defaults < config.yaml < CLI flags
    defaults = {
        "model": "unsloth/Qwen3.5-0.8B",
        "dataset": "data/hybrid_living_agent.jsonl",
        "output": "checkpoints/living-agent",
        "seq_length": 16384,
        "epochs": 1,
        "batch_size": 1,
        "grad_accum": 8,
        "lr": 2e-4,
        "lora_rank": 64,
        "lora_alpha": 128,
        "eval_split": 0.02,
        "save_steps": 200,
        "packing": False,
        "export_gguf": False,
        "export_merged": False,
        "profile_tensors": False,
        "profile_deep": False,
        "offload_layers": -1,
        "warmup_ratio": 0.05,
        "mask_tool_results": True,
        "quant_embed": "auto",          # "auto" = True for 4B+, False for smaller
        "quant_lmhead_fp8": "auto",     # "auto" = True for 4B+, False for smaller
        "patch_rmsnorm_bf16": "auto",   # "auto" = True for Qwen3.5
    }

    config = load_config_file(cli_args.config)
    final = dict(defaults)
    for k, v in config.items():
        if v is not None:
            final[k] = v
    cli_dict = vars(cli_args)
    for cli_key, val in cli_dict.items():
        if cli_key == "config":
            continue
        cfg_key = cli_key.replace("-", "_")
        if val is not None:
            final[cfg_key] = val
    # Handle --no-* flags (override --* flags and config)
    if cli_args.no_quant_embed:
        final["quant_embed"] = False
    elif cli_args.quant_embed:
        final["quant_embed"] = True
    if cli_args.no_quant_lmhead_fp8:
        final["quant_lmhead_fp8"] = False
    elif cli_args.quant_lmhead_fp8:
        final["quant_lmhead_fp8"] = True
    if cli_args.no_patch_rmsnorm_bf16:
        final["patch_rmsnorm_bf16"] = False
    elif cli_args.patch_rmsnorm_bf16:
        final["patch_rmsnorm_bf16"] = True
    # Clean up the no_ keys from final
    for k in ["no_quant_embed", "no_quant_lmhead_fp8", "no_patch_rmsnorm_bf16"]:
        final.pop(k, None)

    args = argparse.Namespace(**final)

    # Resolve "auto" settings based on model size
    # Heuristic: models with "0.8B" or "2B" in the name are small, skip heavy quant
    _model_name_lower = args.model.lower()
    _is_small_model = any(s in _model_name_lower for s in ["0.8b", "0.6b", "1b", "2b", "1.5b"])
    if args.quant_embed == "auto":
        args.quant_embed = not _is_small_model
    if args.quant_lmhead_fp8 == "auto":
        args.quant_lmhead_fp8 = not _is_small_model
    if args.patch_rmsnorm_bf16 == "auto":
        args.patch_rmsnorm_bf16 = "qwen3" in _model_name_lower or "qwen3.5" in _model_name_lower

    print(f"{'='*60}")
    print(f"Living Agent Fine-tuning (Thinker LoRA)")
    print(f"{'='*60}")
    print(f"  Model:        {args.model}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Seq length:   {args.seq_length}")
    print(f"  LoRA:         rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch:        {args.batch_size} x {args.grad_accum} grad accum")
    print(f"  LR:           {args.lr}")
    print(f"  Packing:      {args.packing}")
    print(f"  Mask results: {args.mask_tool_results}")
    print(f"  Quant embed:  {args.quant_embed}")
    print(f"  Quant fp8:    {args.quant_lmhead_fp8}")
    print(f"  RMSNorm bf16: {args.patch_rmsnorm_bf16}")
    print(f"  Output:       {args.output}")
    print(f"  CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    if torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:         {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print()

    # ─── Load model ───
    print("Loading model...")
    import transformers.core_model_loading as _cml
    _cml.GLOBAL_WORKERS = 1

    _n_gpus = torch.cuda.device_count()
    _load_kwargs = dict(
        model_name=args.model,
        max_seq_length=args.seq_length,
        load_in_4bit=True,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        unsloth_tiled_mlp=True,
    )
    if _n_gpus > 1:
        print(f"  Multi-GPU: {_n_gpus} GPUs detected (weight offloading mode)")
        for i in range(_n_gpus):
            _name = torch.cuda.get_device_name(i)
            _mem = torch.cuda.get_device_properties(i).total_mem / 1e9
            print(f"    GPU {i}: {_name} ({_mem:.1f} GB)")
    model, tokenizer = FastLanguageModel.from_pretrained(**_load_kwargs)

    # ─── Force SDPA on GQA attention layers ───
    _text_config = getattr(model.config, "text_config", model.config)
    _old_attn = getattr(_text_config, "_attn_implementation", None)
    _text_config._attn_implementation = "sdpa"
    if hasattr(_text_config, "attn_implementation"):
        _text_config.attn_implementation = "sdpa"
    for _m in model.modules():
        if hasattr(_m, "config") and hasattr(_m.config, "_attn_implementation"):
            _m.config._attn_implementation = "sdpa"
    print(f"  Attention: forced SDPA (was {_old_attn!r})")

    try:
        import fla
        print("  DeltaNet fast path: ENABLED (flash-linear-attention)")
    except ImportError:
        print("  DeltaNet fast path: DISABLED (will use O(n^2) fallback!)")

    try:
        import flash_attn
        print(f"  Flash Attention 2:  ENABLED (v{flash_attn.__version__})")
    except ImportError:
        print("  Flash Attention 2:  DISABLED (using xformers fallback)")

    # ─── Apply LoRA ───
    print("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ─── Remove vision encoder (unused for text-only training) ───
    for vision_attr in ["visual", "vision_model", "vision_tower", "multi_modal_projector"]:
        for obj_path in [model, getattr(model, "model", None), getattr(getattr(model, "model", None), "model", None)]:
            if obj_path is not None and hasattr(obj_path, vision_attr):
                vis = getattr(obj_path, vision_attr)
                if vis is not None:
                    vis_size = sum(p.nelement() * p.element_size() for p in vis.parameters())
                    setattr(obj_path, vision_attr, None)
                    print(f"  Removed {vision_attr}: freed {vis_size/1e9:.2f} GB")
    gc.collect()
    torch.cuda.empty_cache()

    # ─── Optimize large unquantized tensors ───
    target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    base = None
    for path in [
        lambda m: m.model.model.model.language_model,
        lambda m: m.base_model.model.model.language_model,
        lambda m: m.model.model.language_model,
        lambda m: m.model.language_model,
        lambda m: m.language_model,
        lambda m: m.model.model.model,
        lambda m: m.model.model,
        lambda m: m.model,
        lambda m: m,
    ]:
        try:
            candidate = path(model)
            if hasattr(candidate, "lm_head") and hasattr(candidate, "model"):
                base = candidate
                break
        except (AttributeError, TypeError):
            continue

    embed = None
    if base is not None:
        if base.lm_head.weight.dtype == torch.float32:
            base.lm_head.weight.data = base.lm_head.weight.data.to(target_dtype)
            print(f"  lm_head cast to {target_dtype} (enables Apple CCE)")
        print(f"  lm_head: {base.lm_head.weight.dtype}, requires_grad={base.lm_head.weight.requires_grad}")

        # CCE condition check
        requires_grad_ = base.lm_head.weight.requires_grad
        requires_grad_ = requires_grad_ or base.lm_head.weight.dtype == torch.float32
        UNSLOTH_ENABLE_CCE = os.environ.get("UNSLOTH_ENABLE_CCE", "1") == "1"
        NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
        loss_fn_name = getattr(base, 'loss_function', None)
        if loss_fn_name is not None:
            loss_fn_name = loss_fn_name.__name__
        print(f"  CCE conditions: enable={UNSLOTH_ENABLE_CCE}, not_return={NOT_RETURN_LOGITS}, loss_fn={loss_fn_name}, req_grad={requires_grad_}")
        print(f"  CCE WILL ACTIVATE: {UNSLOTH_ENABLE_CCE and NOT_RETURN_LOGITS and (loss_fn_name or '').endswith('ForCausalLMLoss') and not requires_grad_}")

        # Quantize embed_tokens to int8
        for obj in [base, getattr(base, "model", None)]:
            if obj is None:
                continue
            if hasattr(obj, "embed_tokens"):
                embed = obj.embed_tokens
                break
        if embed is None:
            for name, module in model.named_modules():
                if name.endswith("embed_tokens"):
                    embed = module
                    break

        if embed is not None:
            emb_size = embed.weight.nelement() * embed.weight.element_size()
            print(f"  embed_tokens: {embed.weight.dtype}, {emb_size/1e9:.2f} GB")

            if args.quant_embed and embed.weight.dtype in (torch.bfloat16, torch.float16):
                w = embed.weight.data.clone()
                scale = w.abs().amax(dim=1, keepdim=True) / 127.0
                scale = scale.clamp(min=1e-8)
                int8_w = (w / scale).clamp(-127, 127).to(torch.int8)
                embed._embed_scale = scale.squeeze(1).to("cuda")
                embed._embed_dtype = w.dtype
                embed.weight = torch.nn.Parameter(int8_w, requires_grad=False)
                del w, int8_w
                gc.collect()
                torch.cuda.empty_cache()

                def _quantized_embed_forward(input_ids, _embed=embed):
                    int8_out = torch.nn.functional.embedding(input_ids, _embed.weight.data.to(torch.int16).to(_embed._embed_dtype))
                    scales = _embed._embed_scale[input_ids]
                    return int8_out * scales.unsqueeze(-1)

                embed.forward = _quantized_embed_forward
                new_size = embed.weight.nelement() * embed.weight.element_size() + embed._embed_scale.nelement() * embed._embed_scale.element_size()
                saved = emb_size - new_size
                print(f"  embed_tokens quantized to int8: {new_size/1e9:.2f} GB (saved {saved/1e9:.2f} GB)")
            elif not args.quant_embed:
                print(f"  embed_tokens: skipping int8 quantization (disabled)")

    # ─── Patch RMSNorm to stay in bf16 (optional) ───
    if args.patch_rmsnorm_bf16:
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

            def _bf16_rmsnorm_forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                variance = hidden_states.to(torch.bfloat16).pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
                return (hidden_states * (1.0 + self.weight.to(hidden_states.dtype))).to(input_dtype)

            Qwen3_5RMSNorm.forward = _bf16_rmsnorm_forward
            _norm_count = sum(1 for m in model.modules() if isinstance(m, Qwen3_5RMSNorm))
            print(f"  RMSNorm patched to bf16: {_norm_count} instances")
        except ImportError:
            print("  RMSNorm patch skipped (not Qwen3.5)")
    else:
        print("  RMSNorm bf16 patch: skipped (disabled)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ─── Load dataset ───
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    print(f"  Raw examples: {len(dataset)}")

    # Convert living agent format to ChatML text
    print("Formatting with chat template...")
    dataset = dataset.map(
        lambda ex: format_living_agent_to_chatml(ex, tokenizer),
        remove_columns=dataset.column_names,
        num_proc=4,
        desc="Formatting",
    )

    # Filter empty/invalid
    before = len(dataset)
    dataset = dataset.filter(lambda ex: len(ex["text"]) > 0)
    print(f"  After validity filter: {len(dataset)} (dropped {before - len(dataset)} invalid)")

    # Filter by length
    max_chars = args.seq_length * 4
    before = len(dataset)
    dataset = dataset.filter(lambda ex: len(ex["text"]) <= max_chars)
    print(f"  After length filter: {len(dataset)} (dropped {before - len(dataset)} too-long)")

    # Split
    if args.eval_split > 0:
        split = dataset.train_test_split(test_size=args.eval_split, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"  Train: {len(train_dataset)}, Eval: none")

    # ─── Training ───
    os.makedirs(args.output, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=args.output,
            max_seq_length=args.seq_length,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_steps=10,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            logging_steps=1,
            save_steps=args.save_steps,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=args.save_steps if eval_dataset else None,
            save_total_limit=3,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=42,
            dataset_num_proc=4,
            report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
            run_name=f"living-agent-{args.model.split('/')[-1]}-r{args.lora_rank}-ctx{args.seq_length}",
            packing=args.packing,
        ),
    )

    # ─── Quantize lm_head to fp8 (after SFTTrainer init, optional) ───
    if args.quant_lmhead_fp8 and base is not None and base.lm_head.weight.dtype in (torch.bfloat16, torch.float16) and not base.lm_head.weight.requires_grad:
        lm_size_before = base.lm_head.weight.nelement() * base.lm_head.weight.element_size()
        _old_w = base.lm_head.weight.data
        _fp8_w = _old_w.to(torch.float8_e4m3fn)
        base.lm_head.weight = torch.nn.Parameter(_fp8_w, requires_grad=False)
        del _fp8_w, _old_w
        gc.collect()
        torch.cuda.empty_cache()
        lm_size_after = base.lm_head.weight.nelement() * base.lm_head.weight.element_size()
        print(f"  lm_head quantized to fp8: {lm_size_after/1e9:.2f} GB (saved {(lm_size_before-lm_size_after)/1e9:.2f} GB)")
    elif not args.quant_lmhead_fp8:
        print(f"  lm_head fp8: skipped (disabled)")

    # ─── Fix gradient checkpointing: enable Unsloth CPU offloading ───
    from unsloth_zoo.gradient_checkpointing import unsloth_checkpoint as _unsloth_ckpt
    from transformers import PreTrainedModel
    _orig_set_gc = PreTrainedModel._set_gradient_checkpointing
    def _patched_set_gc(self, enable=True, gradient_checkpointing_func=None):
        _orig_set_gc(self, enable=enable, gradient_checkpointing_func=gradient_checkpointing_func)
        if enable:
            for module in self.modules():
                if hasattr(module, "_gradient_checkpointing_func"):
                    module._gradient_checkpointing_func = _unsloth_ckpt
    PreTrainedModel._set_gradient_checkpointing = _patched_set_gc
    print(f"  Gradient checkpointing: patched → Unsloth CPU offloading")

    # ─── Tool result masking ───
    tool_result_masker = None
    if args.mask_tool_results:
        tool_result_masker = LivingAgentDataCollator(tokenizer)
        print(f"  Tool result masking: ENABLED (loss computed only on model-generated tokens)")

        # Wrap the trainer's compute_loss to apply tool_result masking
        _original_compute_loss = trainer.compute_loss

        def _masked_compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
            if "labels" in inputs and tool_result_masker is not None:
                inputs["labels"] = tool_result_masker.mask_tool_results_in_labels(
                    inputs["labels"], inputs["input_ids"]
                )
            if num_items_in_batch is not None:
                return _original_compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
            return _original_compute_loss(model, inputs, return_outputs=return_outputs)

        trainer.compute_loss = _masked_compute_loss

    # ─── Multi-GPU weight offloading ───
    _lm_head_offloaded = False
    _embed_offloaded = False
    if _n_gpus > 1:
        _gpu0_mem = torch.cuda.get_device_properties(0).total_mem
        _gpu1_mem = torch.cuda.get_device_properties(1).total_mem
        if _gpu1_mem > _gpu0_mem:
            print(f"  WARNING: GPU 1 larger than GPU 0! Swap CUDA_VISIBLE_DEVICES.")
            _n_gpus = 1

    if _n_gpus > 1:
        _offload_dev = torch.device("cuda:1")
        _exec_dev = torch.device("cuda:0")
        _gpu1_capacity = torch.cuda.get_device_properties(1).total_mem
        _gpu1_used = 0
        _gpu1_limit = int(_gpu1_capacity * 0.90)
        _offloaded_bytes = 0
        _offloaded_count = 0

        _layers_obj = model
        for _attr in ["model", "base_model", "model", "model", "language_model", "model"]:
            if hasattr(_layers_obj, _attr):
                _layers_obj = getattr(_layers_obj, _attr)

        def _move_layer(layer, device):
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    continue
                param.data = param.data.to(device, non_blocking=True)
                qs = getattr(param, "quant_state", None)
                if qs is not None:
                    qs.to(device)

        _last_loaded = [None]

        def _setup_offload_hooks(layer, storage_dev, compute_dev):
            def _pre_fwd(module, inputs):
                if _last_loaded[0] is not None and _last_loaded[0] is not module:
                    _move_layer(_last_loaded[0], storage_dev)
                _move_layer(module, compute_dev)
                _last_loaded[0] = module
            layer.register_forward_pre_hook(_pre_fwd)

        if hasattr(_layers_obj, "layers"):
            _decoder_layers = _layers_obj.layers
            n_layers = len(_decoder_layers)
            _max_offload = args.offload_layers if args.offload_layers >= 0 else n_layers
            print(f"\n  Weight offloading to GPU 1:")

            for li in range(n_layers):
                if _offloaded_count >= _max_offload:
                    break
                _layer_bytes = sum(
                    p.nelement() * p.element_size()
                    for p in _decoder_layers[li].parameters()
                    if p.device.type != "meta"
                )
                if _gpu1_used + _layer_bytes > _gpu1_limit:
                    break
                _move_layer(_decoder_layers[li], _offload_dev)
                _setup_offload_hooks(_decoder_layers[li], _offload_dev, _exec_dev)
                _gpu1_used += _layer_bytes
                _offloaded_bytes += _layer_bytes
                _offloaded_count += 1

            torch.cuda.empty_cache()
            print(f"    Offloaded {_offloaded_count}/{n_layers} layers ({_offloaded_bytes/1e9:.2f} GB)")

        # Offload embed_tokens to GPU 1
        if embed is not None:
            embed.weight.data = embed.weight.data.to(_offload_dev, non_blocking=True)
            if hasattr(embed, '_embed_scale'):
                embed._embed_scale = embed._embed_scale.to(_offload_dev, non_blocking=True)
            _embed_offloaded = True

            def _embed_pre_hook(module, args, _dev=_offload_dev, _exec=_exec_dev):
                if module.weight.device != _dev:
                    module.weight.data = module.weight.data.to(_dev)
                    if hasattr(module, '_embed_scale'):
                        module._embed_scale = module._embed_scale.to(_dev)
                    torch.cuda.empty_cache()
                input_ids = args[0]
                if input_ids.device != _dev:
                    return (input_ids.to(_dev),) + args[1:]
                return args

            def _embed_post_hook(module, args, output, _dev=_exec_dev):
                if output.device != _dev:
                    return output.to(_dev)
                return output

            embed.register_forward_pre_hook(_embed_pre_hook)
            embed.register_forward_hook(_embed_post_hook)
            torch.cuda.empty_cache()
            print(f"    embed_tokens → GPU 1")

    # Report memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\n  VRAM before training:")
        print(f"    Allocated: {alloc:.2f} GB")
        print(f"    Reserved:  {reserved:.2f} GB")

    # ─── Profiled training step ───
    original_training_step = trainer.training_step
    _step_count = [0]

    def gpu_nvidia_smi_mb():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True
            )
            return int(out.strip().split("\n")[0])
        except Exception:
            return -1

    def mem_mb():
        return torch.cuda.memory_allocated() / 1e6

    def peak_mb():
        return torch.cuda.max_memory_allocated() / 1e6

    def profiled_training_step(model, inputs, num_items_in_batch=None):
        step = _step_count[0]
        _step_count[0] += 1

        if step > 0:
            torch.cuda.empty_cache()

        seq_len = 0
        if hasattr(inputs, 'keys'):
            for key in ["input_ids", "labels", "attention_mask"]:
                if key in inputs and hasattr(inputs[key], 'shape'):
                    seq_len = inputs[key].shape[-1] if inputs[key].dim() > 1 else inputs[key].shape[0]
                    break

        if step < 3:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = mem_mb()

            loss = original_training_step(model, inputs, num_items_in_batch)

            torch.cuda.synchronize()
            after = mem_mb()
            step_peak = peak_mb()

            print(f"\n  === STEP {step} (seq_len={seq_len}) ===")
            print(f"  Torch: alloc={before:.0f}->{after:.0f}MB  peak={step_peak:.0f}MB  delta={step_peak-before:.0f}MB")

            try:
                import wandb as _wb
                if _wb.run is not None:
                    _wb.log({"seq_len": seq_len, "peak_vram_mb": step_peak}, commit=False)
            except Exception:
                pass
            return loss
        else:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            loss = original_training_step(model, inputs, num_items_in_batch)
            step_peak = peak_mb()
            try:
                import wandb as _wb
                if _wb.run is not None:
                    _wb.log({"seq_len": seq_len, "peak_vram_mb": step_peak}, commit=False)
            except Exception:
                pass
            return loss

    trainer.training_step = profiled_training_step

    print(f"\nStarting training...")
    print(f"  Total steps: ~{len(train_dataset) * args.epochs // (args.batch_size * args.grad_accum)}")
    result = trainer.train()

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Loss:    {result.training_loss:.4f}")
    print(f"  Runtime: {result.metrics['train_runtime']:.0f}s")
    print(f"  Samples/sec: {result.metrics['train_samples_per_second']:.2f}")

    # ─── Save ───
    lora_dir = os.path.join(args.output, "lora")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"  LoRA adapter saved to: {lora_dir}")

    if args.export_gguf:
        print("Exporting GGUF (Q4_K_M)...")
        gguf_dir = os.path.join(args.output, "gguf")
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
        print(f"  GGUF saved to: {gguf_dir}")

    if args.export_merged:
        print("Exporting merged 16-bit model...")
        merged_dir = os.path.join(args.output, "merged")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"  Merged model saved to: {merged_dir}")

    print(f"\nDone! Your living agent LoRA is at: {lora_dir}")


if __name__ == "__main__":
    main()
