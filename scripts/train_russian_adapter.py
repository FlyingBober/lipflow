"""Train or fine-tune a Russian Cyrillic visual speech recognition (VSR) adapter.

This script fine-tunes a CTC projection layer on top of the frozen/trainable
Auto-AVSR 3D-ResNet visual frontend using local Russian practice clips or video datasets.

Usage:
    uv run python scripts/train_russian_adapter.py --clips-dir ~/.local/share/lipflow/clips/onboarding/ru --output models/ru/vsr
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Russian Cyrillic alphabet (33 letters) + space + special tokens
CYRILLIC_LETTERS = [
    "а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й",
    "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф",
    "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я"
]
VOCABULARY = ["<blank>", "<space>"] + CYRILLIC_LETTERS + ["<eos>"]
CHAR_TO_ID = {c: i for i, c in enumerate(VOCABULARY)}
CHAR_TO_ID[" "] = CHAR_TO_ID["<space>"]


def encode_text(text: str) -> list[int]:
    """Encode Russian string to token IDs."""
    text = text.lower().strip()
    return [CHAR_TO_ID[c] for c in text if c in CHAR_TO_ID]


class RussianClipDataset(Dataset):
    def __init__(self, clip_files: list[str]):
        self.files = clip_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        rois = data["rois"]  # (T, 96, 96) uint8
        text = str(data["text"])
        target = encode_text(text)
        return torch.from_numpy(rois), torch.tensor(target, dtype=torch.long)


def collate_fn(batch):
    rois_list, targets_list = zip(*batch)
    lengths = [len(r) for r in rois_list]
    target_lengths = [len(t) for t in targets_list]
    max_len = max(lengths)
    c, h, w = 1, 88, 88
    # Center crop 88x88 and normalize
    padded_rois = torch.zeros(len(batch), max_len, c, h, w, dtype=torch.float32)
    for i, r in enumerate(rois_list):
        f = r.float() / 255.0
        o = (f.shape[-1] - 88) // 2
        crop = f[:, o:o + 88, o:o + 88].unsqueeze(1)
        crop = (crop - 0.421) / 0.165
        padded_rois[i, :len(r)] = crop

    padded_targets = nn.utils.rnn.pad_sequence(targets_list, batch_first=True, padding_value=0)
    return padded_rois, torch.tensor(lengths), padded_targets, torch.tensor(target_lengths)


class RussianCTCModel(nn.Module):
    def __init__(self, vocab_size: int = len(VOCABULARY), d_model: int = 512):
        super().__init__()
        from lipflow.vsr import MODELS, LipReader
        # Re-use the existing visual encoder from the base LipReader
        base = LipReader(device="cpu")
        self.encoder = base.model.encoder
        # Freeze lower layers, train top conformer blocks and new Cyrillic CTC head
        self.ctc_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, lengths):
        # x: (B, T, C, H, W) -> transpose for 3D CNN: (B, C, T, H, W)
        x = x.transpose(1, 2)
        enc, _ = self.encoder(x, lengths)
        logits = self.ctc_head(enc)
        return logits


def train(clips_dir: str, output_dir: str, epochs: int = 25, lr: float = 1e-4, batch_size: int = 4):
    clips = sorted(glob.glob(os.path.join(clips_dir, "*.npz")))
    if not clips:
        print(f"[train] Error: No .npz clips found in {clips_dir}")
        print("Record practice clips first: uv run lipflow onboard or save onboarding clips.")
        return

    print(f"[train] Found {len(clips)} Russian clips in {clips_dir}")
    os.makedirs(output_dir, exist_ok=True)

    dataset = RussianClipDataset(clips)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Training on {device}...")

    model = RussianCTCModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for rois, lengths, targets, target_lengths in loader:
            rois = rois.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(rois, lengths)  # (B, T, V)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, V)
            loss = ctc_loss(log_probs, targets, lengths, target_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"[train] Epoch {epoch + 1}/{epochs} - CTC Loss: {avg_loss:.4f}")

    # Save Russian adapter weights and config
    save_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), save_path)
    vocab_path = os.path.join(output_dir, "units.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for v in VOCABULARY:
            f.write(f"{v}\n")
    print(f"[train] Russian adapter saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", default=os.path.expanduser("~/.local/share/lipflow/clips/onboarding/ru"))
    parser.add_argument("--output", default="models/ru/vsr")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    train(args.clips_dir, args.output, args.epochs, args.lr, args.batch_size)


if __name__ == "__main__":
    main()
