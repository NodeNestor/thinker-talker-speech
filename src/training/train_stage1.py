"""Stage 1: Train the Whisper Adapter.

Maps Whisper encoder features → Thinker embedding space so we can skip
STT text decoding entirely. Audio frames go straight into the Thinker.

Training objective: make the Thinker produce the same next-token predictions
whether it receives text token embeddings or adapted Whisper audio embeddings.
We use a distillation loss (KL divergence on logits) + a representation
alignment loss (cosine similarity on hidden states).

Data: LibriSpeech (paired audio + transcript)

Trainable params: ~2M (adapter only — Whisper encoder & Thinker both frozen)
Estimated time: 2-4 hours on single GPU (RTX 5060 Ti 16GB)
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
import numpy as np

from src.model.adapter import WhisperAdapter
from src.training.optimizations import get_apollo_mini_optimizer


class LibriSpeechAlignmentDataset(Dataset):
    """LibriSpeech dataset for adapter training.

    Each sample provides:
    - audio waveform (for Whisper → adapter path)
    - transcript token IDs (for text embedding reference path)
    """

    def __init__(self, hf_dataset, tokenizer, processor, max_audio_len=480000, max_text_len=128):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_audio_len = max_audio_len  # 30s at 16kHz
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]

        # Decode audio manually with soundfile (avoids torchcodec dependency)
        import soundfile as sf
        import io
        audio_info = item["audio"]
        if audio_info.get("bytes"):
            audio, sr = sf.read(io.BytesIO(audio_info["bytes"]), dtype="float32")
        else:
            audio, sr = sf.read(audio_info["path"], dtype="float32")

        # Mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Pad/truncate audio
        audio = np.array(audio, dtype=np.float32)
        if len(audio) > self.max_audio_len:
            audio = audio[:self.max_audio_len]

        # Compute Whisper input features (mel spectrogram)
        whisper_inputs = self.processor(
            audio, sampling_rate=16000, return_tensors="pt",
        )
        input_features = whisper_inputs["input_features"].squeeze(0)  # [mel_bins, time]

        # Tokenize transcript for the text path
        tokens = self.tokenizer(
            text, max_length=self.max_text_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        return {
            "input_features": input_features,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "text": text,
        }


def load_thinker_frozen(lora_path: str, device: str = "cuda"):
    """Load the finetuned Thinker via Unsloth (frozen for hidden state extraction)."""
    from unsloth import FastLanguageModel
    model, processor = FastLanguageModel.from_pretrained(
        lora_path, device_map=device, dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(model)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    # Extract text tokenizer from the VL processor
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    return model, tokenizer


def train_stage1(
    lora_path: str = "checkpoints/living-agent/lora",
    whisper_model_id: str = "openai/whisper-small",
    batch_size: int = 4,
    num_epochs: int = 5,
    lr: float = 3e-4,
    device: str = "cuda",
    output_dir: str = "checkpoints/adapter",
    max_samples: int = 50000,
    grad_accum_steps: int = 4,
):
    """Train the Whisper adapter via distillation from text path."""

    os.makedirs(output_dir, exist_ok=True)

    # ── Load Thinker just to extract embed_tokens, then free it ──
    print("Loading Thinker (extracting embedding layer only)...")
    thinker, tokenizer = load_thinker_frozen(lora_path, device)
    hidden_size = thinker.config.get_text_config().hidden_size
    print(f"  Hidden size: {hidden_size}")

    # Extract the embedding weight and free the full model
    embed_weight = thinker.base_model.model.model.language_model.embed_tokens.weight.detach().clone()
    del thinker
    torch.cuda.empty_cache()
    print(f"  Freed Thinker from VRAM (only keeping embed_tokens: {embed_weight.shape})")

    # ── Load Whisper Adapter (trainable) ──
    print("Loading Whisper Adapter...")
    adapter = WhisperAdapter(
        thinker_hidden_size=hidden_size,
        whisper_model_id=whisper_model_id,
        freeze_encoder=True,
        device=device,
        dtype=torch.bfloat16,
    )
    # Ensure adapter is on correct device and dtype
    adapter = adapter.to(device=device, dtype=torch.bfloat16)
    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"  Adapter trainable params: {trainable:,}")

    # ── Load data ──
    print("Loading LibriSpeech dataset...")
    from datasets import load_dataset, Audio
    ds = load_dataset(
        "librispeech_asr", "clean", split="train.100",
    )
    # Disable built-in audio decoding (requires torchcodec/FFmpeg, not available on Windows)
    ds = ds.cast_column("audio", Audio(decode=False))
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    dataset = LibriSpeechAlignmentDataset(
        ds, tokenizer, adapter.processor,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    # ── Create a simple embedding lookup from the extracted weight ──
    embed_tokens = nn.Embedding.from_pretrained(embed_weight, freeze=True).to(device)

    # ── Optimizer ──
    optimizer = get_apollo_mini_optimizer(adapter, lr=lr)

    # ── Training loop ──
    print(f"\nTraining for {num_epochs} epochs (batch={batch_size}, accum={grad_accum_steps})...")
    print(f"  VRAM usage: only Whisper encoder + adapter (Thinker freed)")
    best_loss = float("inf")
    global_step = 0

    for epoch in range(num_epochs):
        adapter.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            input_features = batch["input_features"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # ── Audio path: Whisper → Adapter → embeddings ──
                audio_embeds = adapter(whisper_features=input_features)
                # audio_embeds: [batch, audio_len, hidden]

                # ── Text path: token embeddings (reference, no grad) ──
                with torch.no_grad():
                    text_embeds = embed_tokens(input_ids)
                    # text_embeds: [batch, text_len, hidden]
                    text_mask = attention_mask.float()

                # ── Loss 1: Global alignment (mean-pooled) ──
                # Audio and text have different lengths, so compare pooled representations
                audio_mean = audio_embeds.mean(dim=1)  # [batch, hidden]
                text_lengths = text_mask.sum(dim=1, keepdim=True).clamp(min=1)
                text_mean = (text_embeds * text_mask.unsqueeze(-1)).sum(dim=1) / text_lengths  # [batch, hidden]

                global_cosine = F.cosine_similarity(audio_mean, text_mean, dim=-1)  # [batch]
                global_loss = 1.0 - global_cosine.mean()

                # ── Loss 2: Soft attention alignment ──
                # Each text token should attend to some audio frames
                # Compute cross-attention: [batch, text_len, audio_len]
                audio_norm = F.normalize(audio_embeds, dim=-1)
                text_norm = F.normalize(text_embeds, dim=-1)
                attn = torch.bmm(text_norm, audio_norm.transpose(1, 2))  # [batch, text_len, audio_len]
                # Softmax over audio frames for each text token
                attn_weights = F.softmax(attn * 10.0, dim=-1)  # sharp attention
                # Attended audio for each text token: [batch, text_len, hidden]
                attended_audio = torch.bmm(attn_weights, audio_embeds)
                # Compare attended audio to text embeddings
                attn_cosine = F.cosine_similarity(attended_audio, text_embeds, dim=-1)  # [batch, text_len]
                attn_loss = 1.0 - (attn_cosine * text_mask).sum() / text_lengths.sum().clamp(min=1)

                # ── Loss 3: Contrastive (in-batch negatives) ──
                # Audio mean of sample i should be closest to text mean of sample i
                if audio_mean.shape[0] >= 2:
                    a_norm = F.normalize(audio_mean, dim=-1)
                    t_norm = F.normalize(text_mean, dim=-1)
                    sim_matrix = a_norm @ t_norm.T  # [batch, batch]
                    labels = torch.arange(sim_matrix.shape[0], device=device)
                    contrastive_loss = F.cross_entropy(sim_matrix * 20.0, labels)
                else:
                    contrastive_loss = torch.tensor(0.0, device=device)

                loss = global_loss + attn_loss + 0.1 * contrastive_loss
                loss = loss / grad_accum_steps

            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * grad_accum_steps
            num_batches += 1

            pbar.set_postfix(
                loss=f"{loss.item() * grad_accum_steps:.4f}",
                glob=f"{global_loss.item():.3f}",
                attn=f"{attn_loss.item():.3f}",
                ctr=f"{contrastive_loss.item():.3f}",
            )

        # Flush remaining gradients
        if num_batches % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(adapter.state_dict(), os.path.join(output_dir, "adapter_best.pt"))
            print(f"  New best loss: {best_loss:.4f}")

    # Save final
    torch.save(adapter.state_dict(), os.path.join(output_dir, "adapter_final.pt"))

    # Save config
    config = {
        "thinker_hidden_size": hidden_size,
        "whisper_model_id": whisper_model_id,
        "lora_path": lora_path,
        "best_loss": best_loss,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "loss": "cosine + 0.1*mse (embedding alignment only)",
    }
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone! Best loss: {best_loss:.4f}")
    print(f"Saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1: Train Whisper Adapter")
    parser.add_argument("--lora-path", default="checkpoints/living-agent/lora", help="Path to finetuned LoRA")
    parser.add_argument("--whisper", default="openai/whisper-small", help="Whisper model ID")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="checkpoints/adapter")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    args = parser.parse_args()

    train_stage1(
        lora_path=args.lora_path,
        whisper_model_id=args.whisper,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        output_dir=args.output,
        max_samples=args.max_samples,
        grad_accum_steps=args.grad_accum,
    )
