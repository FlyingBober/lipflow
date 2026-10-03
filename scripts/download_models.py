"""Download Lipflow model weights (~1.2 GB) if not already present."""
from __future__ import annotations

import os
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")

HF = "https://huggingface.co"
MODELS = [
    (f"{HF}/Amanvir/LRS3_V_WER19.1/resolve/main/model.json", "vsr/model.json", 1000),
    (f"{HF}/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth", "vsr/model.pth", 900_000_000),
    (f"{HF}/Amanvir/lm_en_subword/resolve/main/model.json", "lm/model.json", 1000),
    (f"{HF}/Amanvir/lm_en_subword/resolve/main/model.pth", "lm/model.pth", 200_000_000),
    ("https://github.com/mpc001/auto_avsr/raw/main/spm/unigram/unigram5000.model", "lm/unigram5000.model", 100_000),
    ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
     "face_landmarker.task", 3_000_000),
]


def download(url: str, rel_path: str, min_size: int):
    dest = os.path.join(MODELS_DIR, rel_path)
    if os.path.exists(dest) and os.path.getsize(dest) >= min_size:
        print(f"✓ {rel_path}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    print(f"↓ Downloading {rel_path}...")

    def progress(count, block_size, total_size):
        if total_size > 0:
            pct = count * block_size * 100 / total_size
            sys.stdout.write(f"\r  {pct:.1f}% ({count * block_size / 1e6:.1f} MB / {total_size / 1e6:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, part, reporthook=progress)
    sys.stdout.write("\n")
    os.replace(part, dest)
    print(f"✓ Saved {rel_path}")


def main():
    print(f"Checking model weights in {MODELS_DIR}...")
    for url, rel_path, min_size in MODELS:
        download(url, rel_path, min_size)
    print("All models ready.")


if __name__ == "__main__":
    main()
