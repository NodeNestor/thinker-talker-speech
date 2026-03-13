"""Stage 4: Train the Thinker→Talker Connector.

Maps Thinker hidden states + emotion conditioning + speaker embedding
directly into Chatterbox's input space. This skips text re-tokenization
on the output side, letting the LLM's internal representation drive TTS.

Key design: paralinguistic tags like [laugh], [chuckle], [pause] are preserved
through the text path and handled natively by Chatterbox. The connector only
handles the prosody/emotion conditioning channel — both paths work together.

Training objective:
  - Reconstruction loss: connector output → small decoder → predict Chatterbox
    mel spectrogram features from paired (text, audio) data
  - Prosody consistency: emotion probe conditioning should produce prosody
    features consistent with the reference audio's prosody
  - Speaker consistency: connector output with speaker A's embedding should
    be closer to speaker A's real features than speaker B's

Data: LibriSpeech (paired text + audio, multi-speaker)

Trainable params: ~5M (connector only — Thinker, probe, speaker encoder all frozen)
Estimated time: 3-5 hours on single GPU (RTX 5060 Ti 16GB)
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

from src.model.emotion_probe import EmotionProbe, EMOTION_LABELS
from src.model.connector import ThinkerTalkerConnector
from src.training.optimizations import get_apollo_mini_optimizer
from src.training.train_probe import HiddenStateCapture


class MelDecoder(nn.Module):
    """Small decoder head for training the connector.

    Converts connector output → mel spectrogram features.
    This is only used during training as a reconstruction target.
    At inference time, we feed connector output directly into Chatterbox.
    """

    def __init__(self, input_dim: int = 512, mel_bins: int = 80):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, mel_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, seq, input_dim] → [batch, seq, mel_bins]"""
        return self.decoder(x)


class ConnectorDataset(Dataset):
    """Dataset for connector training — needs text + audio pairs.

    For each sample, we:
    1. Tokenize text → feed to Thinker → get hidden states + emotion
    2. Compute target mel spectrogram from audio
    3. Compute speaker embedding from audio
    """

    def __init__(self, hf_dataset, tokenizer, max_text_len=128, max_audio_len=480000):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len

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

        audio = np.array(audio, dtype=np.float32)

        # Truncate audio
        if len(audio) > self.max_audio_len:
            audio = audio[:self.max_audio_len]

        # Tokenize text
        tokens = self.tokenizer(
            text, max_length=self.max_text_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        # Speaker ID for contrastive loss (LibriSpeech has speaker_id)
        speaker_id = item.get("speaker_id", 0)

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "audio": torch.from_numpy(audio),
            "speaker_id": speaker_id,
            "text": text,
        }


class SyntheticConnectorDataset(Dataset):
    """Dataset from our Thinker + Chatterbox synthetic pipeline."""

    def __init__(self, manifest_path, tokenizer, max_text_len=128, max_audio_len=480000):
        import soundfile as sf
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.samples = manifest["samples"]
        self.sample_rate = manifest["sample_rate"]
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len
        # Map speaker refs to IDs for contrastive loss
        speakers = sorted(set(s["speaker_ref"] for s in self.samples))
        self.speaker_map = {s: i for i, s in enumerate(speakers)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import soundfile as sf
        import librosa

        item = self.samples[idx]
        text = item["text"]

        # Load audio (24kHz from Chatterbox)
        audio, sr = sf.read(item["audio_path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz for mel target / speaker encoder
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        audio = np.array(audio, dtype=np.float32)
        if len(audio) > self.max_audio_len:
            audio = audio[:self.max_audio_len]

        # Tokenize text
        tokens = self.tokenizer(
            text, max_length=self.max_text_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        speaker_id = self.speaker_map[item["speaker_ref"]]

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "audio": torch.from_numpy(audio),
            "speaker_id": speaker_id,
            "text": text,
        }


def collate_fn(batch):
    """Custom collate to handle variable-length audio."""
    # Pad audio to max length in batch
    max_audio_len = max(b["audio"].shape[0] for b in batch)

    padded_audio = []
    for b in batch:
        audio = b["audio"]
        if audio.shape[0] < max_audio_len:
            padding = torch.zeros(max_audio_len - audio.shape[0])
            audio = torch.cat([audio, padding])
        padded_audio.append(audio)

    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "audio": torch.stack(padded_audio),
        "speaker_id": torch.tensor([b["speaker_id"] for b in batch]),
    }


def compute_mel_spectrogram(audio: torch.Tensor, n_mels: int = 80, sr: int = 16000) -> torch.Tensor:
    """Compute mel spectrogram as training target.

    Args:
        audio: [batch, samples]

    Returns:
        mel: [batch, time, n_mels]
    """
    import torchaudio
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=320,
        n_mels=n_mels, power=2.0,
    ).to(audio.device)
    mel = mel_transform(audio)  # [batch, n_mels, time]
    mel = torch.log1p(mel)  # log-mel
    return mel.transpose(1, 2)  # [batch, time, n_mels]


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


def train_stage4(
    lora_path: str = "checkpoints/living-agent/lora",
    probe_checkpoint: str = "checkpoints/probe/probe_best.pt",
    batch_size: int = 4,
    num_epochs: int = 8,
    lr: float = 3e-4,
    device: str = "cuda",
    output_dir: str = "checkpoints/connector",
    max_samples: int = 50000,
    recon_weight: float = 1.0,
    speaker_weight: float = 0.3,
    grad_accum_steps: int = 4,
    synthetic_data: str = None,
):
    """Train the Thinker→Talker connector."""

    os.makedirs(output_dir, exist_ok=True)

    # ── Load Thinker (frozen, via Unsloth) ──
    print("Loading Thinker (frozen, via Unsloth)...")
    thinker, tokenizer = load_thinker_frozen(lora_path, device)
    hidden_size = thinker.config.get_text_config().hidden_size
    print(f"  Hidden size: {hidden_size}")

    # Register hooks to capture hidden states
    capture = HiddenStateCapture(thinker)

    # ── Load Emotion Probe (frozen) ──
    print("Loading Emotion Probe (frozen)...")
    probe = EmotionProbe(hidden_size=hidden_size).to(device)
    if os.path.exists(probe_checkpoint):
        probe.load_state_dict(torch.load(probe_checkpoint, map_location=device))
        print(f"  Loaded probe from {probe_checkpoint}")
    else:
        print(f"  WARNING: Probe checkpoint not found at {probe_checkpoint}")
        print(f"  Using random probe weights (train Stage 3 first for best results)")
    probe.eval()
    for param in probe.parameters():
        param.requires_grad = False

    # ── Load Speaker Encoder (frozen) ──
    print("Loading Speaker Encoder (frozen)...")
    try:
        from src.model.speaker_encoder import SpeakerEncoder
        speaker_enc = SpeakerEncoder(device=device)
        has_speaker_enc = True
    except Exception as e:
        print(f"  WARNING: Speaker encoder failed to load: {e}")
        print(f"  Using random speaker embeddings as placeholder")
        has_speaker_enc = False

    # ── Create Connector (trainable) + MelDecoder (trainable) ──
    print("Creating Connector...")
    connector = ThinkerTalkerConnector(
        thinker_hidden_size=hidden_size,
        talker_hidden_size=512,
        emotion_dim=len(EMOTION_LABELS) + 4,  # 10 emotions + 4 prosody = 14
        speaker_dim=192,  # ECAPA-TDNN
    ).to(device)

    mel_decoder = MelDecoder(input_dim=512, mel_bins=80).to(device)

    trainable_params = (
        sum(p.numel() for p in connector.parameters() if p.requires_grad)
        + sum(p.numel() for p in mel_decoder.parameters() if p.requires_grad)
    )
    print(f"  Trainable params: {trainable_params:,} (connector + mel decoder)")

    # ── Load data ──
    if synthetic_data:
        print(f"Loading synthetic dataset from {synthetic_data}...")
        dataset = SyntheticConnectorDataset(synthetic_data, tokenizer)
        print(f"  {len(dataset)} samples")
    else:
        print("Loading LibriSpeech dataset...")
        from datasets import load_dataset, Audio
        ds = load_dataset(
            "librispeech_asr", "clean", split="train.100",
        )
        ds = ds.cast_column("audio", Audio(decode=False))
        if max_samples and len(ds) > max_samples:
            ds = ds.select(range(max_samples))
        dataset = ConnectorDataset(ds, tokenizer)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
        collate_fn=collate_fn,
    )

    # ── Optimizer (covers both connector + mel_decoder) ──
    all_trainable = nn.ModuleList([connector, mel_decoder])
    optimizer = get_apollo_mini_optimizer(all_trainable, lr=lr)

    # ── Training loop ──
    print(f"\nTraining for {num_epochs} epochs (batch={batch_size}, accum={grad_accum_steps})...")
    best_loss = float("inf")
    global_step = 0

    for epoch in range(num_epochs):
        connector.train()
        mel_decoder.train()
        total_loss = 0
        total_recon = 0
        total_speaker = 0
        total_duration = 0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio = batch["audio"].to(device)
            speaker_ids = batch["speaker_id"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # ── Step 1: Thinker forward (frozen) → hidden states ──
                capture.clear()
                with torch.no_grad():
                    out = thinker(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=True)
                    # Get last hidden state from model output
                    if hasattr(out, 'hidden_states') and out.hidden_states:
                        thinker_hidden = out.hidden_states[-1]
                    else:
                        # Fallback: use the last captured state
                        all_states = {**capture.deltanet_states, **capture.attention_states}
                        if all_states:
                            thinker_hidden = all_states[max(all_states.keys())]
                        else:
                            continue  # skip batch if no states captured

                # ── Step 2: Emotion probe (frozen) → conditioning vector ──
                with torch.no_grad():
                    probe_out = probe(capture.deltanet_states, capture.attention_states)
                    emotion_vector = probe_out["conditioning_vector"]  # [batch, 14]

                # ── Step 3: Speaker embedding (frozen) ──
                with torch.no_grad():
                    if has_speaker_enc:
                        speaker_emb = speaker_enc(audio, sample_rate=16000)  # [batch, 192]
                    else:
                        speaker_emb = torch.randn(input_ids.shape[0], 192, device=device)

                # ── Step 4: Compute target mel FIRST (need length for upsampling) ──
                with torch.no_grad():
                    target_mel = compute_mel_spectrogram(audio)  # [batch, time, 80]
                target_len = target_mel.shape[1]

                # ── Step 5: Connector forward (trainable) — upsample to mel rate ──
                connector_out, predicted_durations = connector.forward_with_durations(
                    thinker_hidden=thinker_hidden,
                    emotion_vector=emotion_vector,
                    speaker_embedding=speaker_emb,
                    target_length=target_len,
                )
                # connector_out: [batch, target_len, 512]

                # ── Step 6: Mel decoder (trainable) → predict mel ──
                predicted_mel = mel_decoder(connector_out)  # [batch, target_len, 80]

                # ── Loss 1: Reconstruction (MSE on mel) ──
                recon_loss = F.mse_loss(predicted_mel, target_mel)

                # ── Loss 2: Duration prediction ──
                # Target: uniform distribution of frames across tokens
                seq_len = thinker_hidden.shape[1]
                target_dur = torch.full(
                    (input_ids.shape[0], seq_len),
                    target_len / seq_len,
                    device=device,
                )
                duration_loss = F.mse_loss(predicted_durations, target_dur)

                # ── Loss 2: Speaker consistency (contrastive) ──
                # Connector outputs for same speaker should be closer
                speaker_loss = torch.tensor(0.0, device=device)
                if speaker_weight > 0 and input_ids.shape[0] >= 4:
                    # Pool connector output across time
                    pooled = connector_out.mean(dim=1)  # [batch, 512]
                    pooled_norm = F.normalize(pooled, dim=-1)

                    # Cosine similarity matrix
                    sim_matrix = pooled_norm @ pooled_norm.T  # [batch, batch]

                    # Labels: 1 if same speaker, 0 otherwise
                    speaker_labels = (speaker_ids.unsqueeze(0) == speaker_ids.unsqueeze(1)).float()
                    # Remove diagonal
                    mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=device)
                    sim_flat = sim_matrix[mask]
                    label_flat = speaker_labels[mask]

                    # Binary cross-entropy: same speaker → high similarity
                    speaker_loss = F.binary_cross_entropy_with_logits(
                        sim_flat * 5.0,  # scale for sharper predictions
                        label_flat,
                    )

                # ── Combined loss ──
                duration_weight = 0.1
                loss = recon_weight * recon_loss + speaker_weight * speaker_loss + duration_weight * duration_loss
                loss = loss / grad_accum_steps

            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(connector.parameters()) + list(mel_decoder.parameters()),
                    1.0,
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * grad_accum_steps
            total_recon += recon_loss.item()
            total_speaker += speaker_loss.item()
            total_duration += duration_loss.item()
            num_batches += 1

            pbar.set_postfix(
                loss=f"{loss.item() * grad_accum_steps:.4f}",
                recon=f"{recon_loss.item():.4f}",
                spk=f"{speaker_loss.item():.4f}",
                dur=f"{duration_loss.item():.4f}",
            )

        # Flush remaining gradients
        if num_batches % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                list(connector.parameters()) + list(mel_decoder.parameters()),
                1.0,
            )
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_speaker = total_speaker / num_batches
        avg_duration = total_duration / num_batches
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} (recon={avg_recon:.4f}, spk={avg_speaker:.4f}, dur={avg_duration:.4f})")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(connector.state_dict(), os.path.join(output_dir, "connector_best.pt"))
            torch.save(mel_decoder.state_dict(), os.path.join(output_dir, "mel_decoder_best.pt"))
            print(f"  New best loss: {best_loss:.4f}")

    # Save final
    torch.save(connector.state_dict(), os.path.join(output_dir, "connector_final.pt"))
    torch.save(mel_decoder.state_dict(), os.path.join(output_dir, "mel_decoder_final.pt"))

    # Save config
    # Cleanup hooks
    capture.remove_hooks()

    config = {
        "thinker_hidden_size": hidden_size,
        "talker_hidden_size": 512,
        "emotion_dim": 14,
        "speaker_dim": 192,
        "mel_bins": 80,
        "lora_path": lora_path,
        "probe_checkpoint": probe_checkpoint,
        "best_loss": best_loss,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
    }
    with open(os.path.join(output_dir, "connector_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone! Best loss: {best_loss:.4f}")
    print(f"Saved to {output_dir}/")
    print(f"\nNote: [laugh], [chuckle], [pause] tags are preserved through the text path.")
    print(f"The connector handles prosody/emotion conditioning only.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 4: Train Thinker→Talker Connector")
    parser.add_argument("--lora-path", default="checkpoints/living-agent/lora", help="Path to finetuned LoRA")
    parser.add_argument("--probe-ckpt", default="checkpoints/probe/probe_best.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="checkpoints/connector")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--synthetic-data", default=None,
                        help="Path to synthetic manifest.json (from generate_connector_data.py)")
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
        synthetic_data=args.synthetic_data,
    )
