"""Train or fine-tune a Russian Cyrillic visual speech recognition (VSR) adapter.

This script fine-tunes a CTC projection layer on top of the frozen/trainable
Auto-AVSR 3D-ResNet visual frontend using local Russian practice clips or video datasets.

Usage:
    uv run python scripts/train_russian_adapter.py --clips-dir ~/.local/share/lipflow/clips/onboarding/ru --output models/ru/vsr
"""
from __future__ import annotations

import argparse
import os

from lipflow.train_ru import train_russian, CYRILLIC_LETTERS, VOCABULARY, RussianCTCModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", default=None, help="Directory containing .npz clips (default: auto-detect)")
    parser.add_argument("--output", default="models/ru/vsr", help="Output directory for model weights and units.txt")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate (default: 0.01)")
    args = parser.parse_args()

    train_russian(clips_dir=args.clips_dir, output_dir=args.output, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
