"""Train or fine-tune a Russian Cyrillic visual speech recognition (VSR) adapter."""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from .paths import HOME, personal_vsr
from .practice import CLIPS
from .russian import CYRILLIC_LETTERS

VOCABULARY = ["<blank>", "<space>"] + CYRILLIC_LETTERS + ["<eos>"]
CHAR_TO_ID = {c: i for i, c in enumerate(VOCABULARY)}
CHAR_TO_ID[" "] = CHAR_TO_ID["<space>"]


def encode_text(text: str) -> list[int]:
    """Encode Russian string to token IDs."""
    text = text.lower().strip()
    return [CHAR_TO_ID[c] for c in text if c in CHAR_TO_ID]


class RussianCTCModel(nn.Module):
    def __init__(self, vocab_size: int = len(VOCABULARY), d_model: int = 768):
        super().__init__()
        from .vsr import LipReader
        # Re-use the existing visual encoder from the base LipReader
        base = LipReader(device="cpu", personal=False)
        self.encoder = base.model.encoder
        self.ctc = base.model.ctc
        self.ctc.ctc_lo = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (1, 1, T, 88, 88) -> (1, T, 768)
        enc, _ = self.encoder(x, None)
        logits = self.ctc.ctc_lo(enc)
        return logits


def find_russian_clips(clips_dir: str | None = None) -> list[str]:
    """Find all .npz clips containing Russian Cyrillic text."""
    dirs_to_check = []
    if clips_dir:
        dirs_to_check.append(clips_dir)
    ru_onboarding = os.path.join(CLIPS, "ru")
    if ru_onboarding not in dirs_to_check:
        dirs_to_check.append(ru_onboarding)
    if CLIPS not in dirs_to_check:
        dirs_to_check.append(CLIPS)
    dictations_dir = os.path.join(os.path.dirname(CLIPS), "clips", "dictations")
    if dictations_dir not in dirs_to_check:
        dirs_to_check.append(dictations_dir)
    fallback_dictations = os.path.join(os.path.dirname(CLIPS), "dictations")
    if fallback_dictations not in dirs_to_check:
        dirs_to_check.append(fallback_dictations)

    all_files = []
    for d in dirs_to_check:
        if os.path.isdir(d):
            all_files.extend(sorted(glob.glob(os.path.join(d, "*.npz"))))

    # Deduplicate while preserving order
    seen = set()
    unique_files = []
    for f in all_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    # Filter for clips that have Russian Cyrillic characters
    ru_clips = []
    for p in unique_files:
        try:
            d = np.load(p, allow_pickle=True)
            txt = str(d.get("text", "")).lower()
            if any(c in CYRILLIC_LETTERS for c in txt):
                ru_clips.append(p)
        except Exception:
            pass

    return ru_clips


def train_russian(
    clips_dir: str | None = None,
    output_dir: str | None = None,
    epochs: int = 25,
    lr: float = 1e-2,
    report=None,
) -> dict:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if output_dir is None:
        output_dir = os.path.join(ROOT, "models", "ru", "vsr")
    elif not os.path.isabs(output_dir):
        output_dir = os.path.join(ROOT, output_dir)

    clips = find_russian_clips(clips_dir)
    if not clips:
        msg = (
            "Не найдено тренировочных клипов с русским текстом.\n"
            "Запустите сначала запись клипов:\n"
            "    uv run lipflow onboard --language ru"
        )
        print(f"[train] Error: {msg}")
        return {
            "before": 0.0,
            "after": None,
            "kept": False,
            "clips": 0,
            "note": "Нет русских тренировочных клипов. Запустите: uv run lipflow onboard --language ru",
        }

    print(f"[train] Found {len(clips)} Russian clips for training.")
    os.makedirs(output_dir, exist_ok=True)

    from .vsr import LipReader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Training on {device}...")

    model = RussianCTCModel().to(device)
    for p in model.encoder.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(model.ctc.ctc_lo.parameters(), lr=lr)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    model.train()
    final_loss = 0.0
    for epoch in range(epochs):
        total_loss = 0.0
        n_samples = 0
        for p in clips:
            data = np.load(p, allow_pickle=True)
            rois = data["rois"]
            text = str(data["text"])
            target = encode_text(text)
            if not target:
                continue

            x = LipReader.to_tensor(rois).unsqueeze(0).to(device)  # (1, 1, T, 88, 88)
            T = x.shape[2]

            optimizer.zero_grad()
            logits = model(x)  # (1, T, V)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (T, 1, V)

            targets = torch.tensor(target, dtype=torch.long, device=device).unsqueeze(0)
            input_lengths = torch.tensor([T], dtype=torch.long)
            target_lengths = torch.tensor([len(target)], dtype=torch.long)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_samples += 1

        avg_loss = total_loss / max(n_samples, 1)
        final_loss = avg_loss
        if report:
            pct = int(100 * (epoch + 1) / epochs)
            report(pct, f"Обучение: эпоха {epoch + 1}/{epochs} (CTC Loss: {avg_loss:.4f})")
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"[train] Epoch {epoch + 1}/{epochs} - CTC Loss: {avg_loss:.4f}", flush=True)

    # 1. Save repo Russian adapter weights, vocab, and config
    save_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), save_path)
    vocab_path = os.path.join(output_dir, "units.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for v in VOCABULARY[1:-1]:
            f.write(f"{v}\n")
    base_json = os.path.join(ROOT, "models", "vsr", "model.json")
    if os.path.exists(base_json):
        with open(base_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg[1] = len(VOCABULARY)
        with open(os.path.join(output_dir, "model.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
    print(f"[train] Russian adapter saved to {save_path}")

    # 2. Save personal face model (~/.local/share/lipflow/models/ru/vsr_face.pth)
    pv = personal_vsr("ru")
    os.makedirs(os.path.dirname(pv), exist_ok=True)
    torch.save(model.state_dict(), pv)
    print(f"[train] Personal Russian face model saved to {pv}")

    return {
        "before": 1.0,
        "after": float(final_loss),
        "kept": True,
        "clips": len(clips),
        "note": f"Модель обучена на {len(clips)} клипах (Loss: {final_loss:.4f})",
    }
