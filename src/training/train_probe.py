"""Stage 3: Train the Emotion/Prosody Probe.

This is the most novel part — we train a small probe (~2M params) to read
emotion and prosody from the Thinker's hidden states.

The Thinker is FROZEN during this stage. We only train the probe.

Data: GoEmotions (text, 28 emotions)

Trainable params: ~2M (probe only)
Estimated time: 1-2 hours on single GPU
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import unsloth  # Must be imported before transformers/peft

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json

from src.model.emotion_probe import EmotionProbe, EMOTION_LABELS
from src.training.optimizations import get_apollo_mini_optimizer


# Map GoEmotions 28 labels to our 10 labels
GOEMOTION_MAP = {
    "admiration": "happy", "amusement": "happy", "approval": "happy",
    "caring": "calm", "desire": "excited", "excitement": "excited",
    "gratitude": "happy", "joy": "happy", "love": "happy",
    "optimism": "happy", "pride": "happy", "relief": "calm",
    "anger": "angry", "annoyance": "angry", "disapproval": "angry",
    "disgust": "disgusted", "embarrassment": "sad",
    "disappointment": "sad", "grief": "sad", "remorse": "sad", "sadness": "sad",
    "confusion": "confused", "curiosity": "surprised",
    "fear": "fearful", "nervousness": "fearful",
    "realization": "surprised", "surprise": "surprised",
    "neutral": "neutral",
}


class TextEmotionDataset(Dataset):
    """GoEmotions dataset mapped to our emotion labels."""

    def __init__(self, hf_dataset, tokenizer, max_length=128):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        # GoEmotions label names
        self.go_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness", "surprise",
            "neutral",
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]

        # Tokenize
        tokens = self.tokenizer(
            text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        # Map GoEmotions multi-label to our single dominant emotion
        label_ids = item["labels"]
        if label_ids:
            go_label = self.go_labels[label_ids[0]]
            our_label = GOEMOTION_MAP.get(go_label, "neutral")
        else:
            our_label = "neutral"

        emotion_idx = EMOTION_LABELS.index(our_label)

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "emotion_label": emotion_idx,
        }


class HiddenStateCapture:
    """Captures hidden states from transformer layers via forward hooks.

    Qwen 3.5 alternates: DeltaNet, DeltaNet, DeltaNet, Attention (repeat).
    We capture both types separately for the emotion probe.
    """

    def __init__(self, model):
        self.deltanet_states = {}
        self.attention_states = {}
        self._hooks = []

        # Find the transformer layers
        layers = self._find_layers(model)
        if not layers:
            raise RuntimeError("Could not find transformer layers in model")

        print(f"  Registering hooks on {len(layers)} layers")
        for i, layer in enumerate(layers):
            layer_type = "attention" if (i + 1) % 4 == 0 else "deltanet"

            def make_hook(layer_idx, ltype):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hs = output[0]
                    else:
                        hs = output
                    if ltype == "deltanet":
                        self.deltanet_states[layer_idx] = hs.detach()
                    else:
                        self.attention_states[layer_idx] = hs.detach()
                return hook_fn

            handle = layer.register_forward_hook(make_hook(i, layer_type))
            self._hooks.append(handle)

    def _find_layers(self, model):
        """Navigate through LoRA/PeftModel wrappers to find transformer layers."""
        # Try various paths (order matters — most specific first)
        candidates = [
            lambda: model.base_model.model.model.language_model.layers,  # PeftModel + Qwen3.5
            lambda: model.base_model.model.language_model.model.layers,
            lambda: model.base_model.model.model.layers,
            lambda: model.model.language_model.layers,
            lambda: model.model.language_model.model.layers,
            lambda: model.model.model.layers,
            lambda: model.language_model.model.layers,
            lambda: model.model.layers,
        ]
        for get_layers in candidates:
            try:
                layers = get_layers()
                if len(layers) > 0:
                    return layers
            except (AttributeError, TypeError):
                continue
        return None

    def clear(self):
        self.deltanet_states.clear()
        self.attention_states.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def load_thinker_frozen(lora_path: str, device: str = "cuda"):
    """Load the finetuned Thinker via Unsloth (frozen)."""
    from unsloth import FastLanguageModel
    model, processor = FastLanguageModel.from_pretrained(
        lora_path, device_map=device, dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(model)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    return model, tokenizer


def train_probe(
    lora_path: str = "checkpoints/living-agent/lora",
    batch_size: int = 16,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda",
    output_dir: str = "checkpoints/probe",
    max_samples: int = 50000,
):
    """Train the emotion/prosody probe."""

    os.makedirs(output_dir, exist_ok=True)

    # Load Thinker (frozen, via Unsloth)
    print("Loading Thinker (frozen, via Unsloth)...")
    thinker, tokenizer = load_thinker_frozen(lora_path, device)
    hidden_size = thinker.config.get_text_config().hidden_size
    print(f"  Hidden size: {hidden_size}")

    # Register hooks to capture hidden states
    capture = HiddenStateCapture(thinker)

    # Create probe
    print("Creating Emotion Probe...")
    probe = EmotionProbe(hidden_size=hidden_size).to(device)
    trainable = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(f"  Probe trainable params: {trainable:,}")

    # Load data
    print("Loading GoEmotions dataset...")
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/go_emotions", split="train")
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    dataset = TextEmotionDataset(ds, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                        pin_memory=True, drop_last=True)

    # Optimizer — Apollo Mini for the probe
    optimizer = get_apollo_mini_optimizer(probe, lr=lr)

    # Loss — similarity-weighted soft labels
    # Similar emotions get soft target mass instead of hard 0/1.
    # This way confusing "sad" with "fearful" is penalized less than "sad" with "happy".
    # Groups: {happy, excited, calm}, {sad, fearful, disgusted}, {angry}, {surprised, confused}, {neutral}
    _EMOTION_GROUPS = {
        "happy": "positive", "excited": "positive", "calm": "positive",
        "sad": "negative", "fearful": "negative", "disgusted": "negative",
        "angry": "hostile",
        "surprised": "reactive", "confused": "reactive",
        "neutral": "neutral",
    }
    _smooth = 0.15  # mass redistributed to same-group emotions
    _soft_targets = torch.zeros(len(EMOTION_LABELS), len(EMOTION_LABELS), device=device)
    for i, ei in enumerate(EMOTION_LABELS):
        group_i = _EMOTION_GROUPS[ei]
        same_group = [j for j, ej in enumerate(EMOTION_LABELS) if _EMOTION_GROUPS[ej] == group_i and j != i]
        _soft_targets[i, i] = 1.0 - _smooth
        if same_group:
            for j in same_group:
                _soft_targets[i, j] = _smooth / len(same_group)
        else:
            _soft_targets[i, i] = 1.0  # no group mates, keep hard

    def soft_cross_entropy(logits, labels):
        """Cross-entropy with similarity-weighted soft targets."""
        targets = _soft_targets[labels]  # [batch, n_classes]
        log_probs = F.log_softmax(logits, dim=-1)
        return -(targets * log_probs).sum(dim=-1).mean()

    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    best_acc = 0.0

    for epoch in range(num_epochs):
        probe.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["emotion_label"].to(device)

            # Forward through frozen Thinker to capture hidden states
            capture.clear()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                thinker(input_ids=input_ids, attention_mask=attention_mask)

            # Forward through probe
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = probe(capture.deltanet_states, capture.attention_states)
                loss = soft_cross_entropy(result["emotion_logits"], labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            preds = result["emotion_logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

        epoch_loss = total_loss / len(loader)
        epoch_acc = correct / total
        print(f"  Epoch {epoch+1}: loss={epoch_loss:.4f}, acc={epoch_acc:.3f}")

        # Save best
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(probe.state_dict(), os.path.join(output_dir, "probe_best.pt"))
            print(f"  New best accuracy: {best_acc:.3f}")

    # Cleanup hooks
    capture.remove_hooks()

    # Save final
    torch.save(probe.state_dict(), os.path.join(output_dir, "probe_final.pt"))

    # Save config
    config = {
        "hidden_size": hidden_size,
        "num_emotions": len(EMOTION_LABELS),
        "emotion_labels": EMOTION_LABELS,
        "best_accuracy": best_acc,
        "epochs": num_epochs,
        "lora_path": lora_path,
    }
    with open(os.path.join(output_dir, "probe_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone! Best accuracy: {best_acc:.3f}")
    print(f"Saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", default="checkpoints/living-agent/lora")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="checkpoints/probe")
    parser.add_argument("--max-samples", type=int, default=50000)
    args = parser.parse_args()

    train_probe(
        lora_path=args.lora_path,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        output_dir=args.output,
        max_samples=args.max_samples,
    )
