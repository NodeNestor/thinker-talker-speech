"""Stage 4: Train the HiddenStateConnector.

Maps Thinker hidden states directly into T3's text embedding space,
bypassing text decoding entirely. This preserves emotion/prosody
information that gets lost when decoding to text.

Training approach:
  For each text sample:
    1. Run through Thinker → get hidden states [B, seq_thinker, 1024]
    2. Tokenize with T3's tokenizer → get T3 text token IDs
    3. Look up T3 text embeddings via t3.text_emb(token_ids) [B, seq_t3, 1024]
    4. Run emotion probe → get conditioning vector [B, 14]
    5. Connector projects thinker_hidden → predicted T3 embeddings
    6. Loss: MSE + cosine similarity between predicted and actual T3 embeddings

Both Thinker (Qwen 3.5, 1024-dim) and T3 Turbo (GPT2-medium, 1024-dim) have
the same hidden size, making alignment natural.

Data: GoEmotions (text with emotion labels) — no audio needed.
      We're aligning text representations between two models.

Trainable params: ~2M (connector only — Thinker, probe, T3 all frozen)
Estimated time: 30-60 min on single GPU
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
from src.model.connector import HiddenStateConnector
from src.training.train_probe import HiddenStateCapture


class TextAlignmentDataset(Dataset):
    """Dataset for connector training — text only (no audio needed).

    We align Thinker hidden states with T3 text embeddings using the same text.
    GoEmotions gives us diverse text with emotion labels.
    """

    def __init__(self, hf_dataset, thinker_tokenizer, max_length=64):
        self.data = hf_dataset
        self.tokenizer = thinker_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]

        tokens = self.tokenizer(
            text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "text": text,
        }


def train_stage4(
    lora_path: str = "checkpoints/living-agent/lora",
    probe_checkpoint: str = "checkpoints/probe/probe_best.pt",
    batch_size: int = 8,
    num_epochs: int = 15,
    lr: float = 1e-3,
    device: str = "cuda",
    output_dir: str = "checkpoints/connector",
    max_samples: int = 30000,
    grad_accum_steps: int = 2,
):
    """Train the HiddenStateConnector."""

    os.makedirs(output_dir, exist_ok=True)

    # ── Load Thinker (frozen) ──
    print("Loading Thinker (frozen, via Unsloth)...")
    from unsloth import FastLanguageModel
    thinker, processor = FastLanguageModel.from_pretrained(
        lora_path, device_map=device, dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(thinker)
    thinker.eval()
    for param in thinker.parameters():
        param.requires_grad = False
    thinker_tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    hidden_size = thinker.config.get_text_config().hidden_size
    print(f"  Thinker hidden size: {hidden_size}")

    # Register hooks for hidden state capture
    capture = HiddenStateCapture(thinker)

    # ── Load Emotion Probe (frozen) ──
    print("Loading Emotion Probe (frozen)...")
    probe = EmotionProbe(hidden_size=hidden_size).to(device)
    if os.path.exists(probe_checkpoint):
        probe.load_state_dict(torch.load(probe_checkpoint, map_location=device))
        print(f"  Loaded from {probe_checkpoint}")
    else:
        print(f"  WARNING: No probe checkpoint, using random weights")
    probe.eval()
    for param in probe.parameters():
        param.requires_grad = False

    # ── Load T3 text embedding (frozen) ──
    print("Loading T3 Turbo text embedding (frozen)...")
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    tts = ChatterboxTurboTTS.from_pretrained(device=device)
    t3 = tts.t3
    t3.eval()
    for param in t3.parameters():
        param.requires_grad = False
    t3_tokenizer = tts.tokenizer
    t3_dim = t3.dim
    print(f"  T3 text_emb: {t3.text_emb.weight.shape}")  # [50276, 1024]
    print(f"  T3 hidden dim: {t3_dim}")

    # ── Create Connector (trainable) ──
    print("Creating HiddenStateConnector...")
    connector = HiddenStateConnector(
        thinker_dim=hidden_size,
        t3_dim=t3_dim,
        emotion_dim=len(EMOTION_LABELS) + 4,  # 10 emotions + 4 prosody = 14
    ).to(device)
    trainable = sum(p.numel() for p in connector.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    # ── Load data ──
    print("Loading GoEmotions dataset...")
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    print(f"  {len(ds)} samples")

    dataset = TextAlignmentDataset(ds, thinker_tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                       num_workers=0, pin_memory=True, drop_last=True)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(connector.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ── Training loop ──
    print(f"\nTraining for {num_epochs} epochs (batch={batch_size}, accum={grad_accum_steps})...")
    best_loss = float("inf")

    for epoch in range(num_epochs):
        connector.train()
        total_loss = 0
        total_mse = 0
        total_cos = 0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            texts = batch["text"]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Step 1: Thinker forward → hidden states
                capture.clear()
                with torch.no_grad():
                    out = thinker(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 output_hidden_states=True)
                    if hasattr(out, 'hidden_states') and out.hidden_states:
                        thinker_hidden = out.hidden_states[-1].float()
                    else:
                        all_states = {**capture.deltanet_states, **capture.attention_states}
                        if not all_states:
                            continue
                        thinker_hidden = all_states[max(all_states.keys())].float()

                # Step 2: Emotion probe → conditioning vector
                with torch.no_grad():
                    delta_f32 = {k: v.float() for k, v in capture.deltanet_states.items()}
                    attn_f32 = {k: v.float() for k, v in capture.attention_states.items()}
                    probe_out = probe(delta_f32, attn_f32)
                    emotion_vector = probe_out["conditioning_vector"].float()

                # Step 3: T3 text embeddings (target)
                with torch.no_grad():
                    t3_tokens = t3_tokenizer(
                        list(texts),
                        max_length=64, truncation=True, padding="max_length",
                        return_tensors="pt",
                    ).input_ids.to(device)
                    target_emb = t3.text_emb(t3_tokens).float()  # [B, seq_t3, 1024]

                target_len = target_emb.shape[1]

                # Step 4: Connector forward (trainable)
                predicted_emb = connector(
                    thinker_hidden, emotion_vector, target_length=target_len,
                )

                # Step 5: Loss
                losses = connector.compute_loss(predicted_emb, target_emb)
                loss = losses["total_loss"] / grad_accum_steps

            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(connector.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += losses["total_loss"].item()
            total_mse += losses["mse_loss"].item()
            total_cos += losses["cosine_loss"].item()
            num_batches += 1

            pbar.set_postfix(
                loss=f"{losses['total_loss'].item():.4f}",
                mse=f"{losses['mse_loss'].item():.4f}",
                cos=f"{losses['cosine_loss'].item():.4f}",
            )

        # Flush remaining gradients
        if num_batches % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(connector.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_cos = total_cos / num_batches
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} (mse={avg_mse:.4f}, cos={avg_cos:.4f})")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(connector.state_dict(),
                       os.path.join(output_dir, "connector_best.pt"))
            print(f"  New best loss: {best_loss:.4f}")

    # Cleanup
    capture.remove_hooks()

    # Save final
    torch.save(connector.state_dict(),
               os.path.join(output_dir, "connector_final.pt"))

    config = {
        "type": "HiddenStateConnector",
        "thinker_dim": hidden_size,
        "t3_dim": t3_dim,
        "emotion_dim": 14,
        "lora_path": lora_path,
        "probe_checkpoint": probe_checkpoint,
        "best_loss": best_loss,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "trainable_params": trainable,
    }
    with open(os.path.join(output_dir, "connector_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone! Best loss: {best_loss:.4f}")
    print(f"Saved to {output_dir}/")
    print(f"\nThe connector projects Thinker hidden states directly into")
    print(f"T3's text embedding space — no text decoding needed at inference.")
    print(f"\nUsage:")
    print(f"  connector = HiddenStateConnector()")
    print(f"  connector.load_state_dict(torch.load('{output_dir}/connector_best.pt'))")
    print(f"  t3_emb = connector(thinker_hidden, emotion_vector)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 4: Train HiddenStateConnector")
    parser.add_argument("--lora-path", default="checkpoints/living-agent/lora")
    parser.add_argument("--probe-ckpt", default="checkpoints/probe/probe_best.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="checkpoints/connector")
    parser.add_argument("--max-samples", type=int, default=30000)
    parser.add_argument("--grad-accum", type=int, default=2)
    args = parser.parse_args()

    train_stage4(
        lora_path=args.lora_path,
        probe_checkpoint=args.probe_ckpt,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        output_dir=args.output,
        max_samples=args.max_samples,
        grad_accum_steps=args.grad_accum,
    )
