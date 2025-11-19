"""
Create a tiny dummy dataset for smoke testing the pipeline.
Usage:
    python scripts/make_dummy_data.py --out_dir data/processed/debug --num 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_sample(img_path: Path, mask_path: Path, size: int = 256):
    rng = np.random.default_rng()
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    # random ellipse as fake polyp
    cx, cy = rng.integers(size // 4, 3 * size // 4, size=2)
    rx, ry = rng.integers(size // 8, size // 5, size=2)
    bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
    draw.ellipse(bbox, fill=255)

    img.save(img_path)
    mask.save(mask_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/processed/debug")
    parser.add_argument("--num", type=int, default=4)
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    train_img = out_root / "train" / "images"
    train_mask = out_root / "train" / "masks"
    val_img = out_root / "val" / "images"
    val_mask = out_root / "val" / "masks"
    for d in [train_img, train_mask, val_img, val_mask]:
        d.mkdir(parents=True, exist_ok=True)

    # split num: half train, half val
    n_train = max(1, args.num // 2)
    n_val = max(1, args.num - n_train)

    for i in range(n_train):
        make_sample(train_img / f"sample_{i}.jpg", train_mask / f"sample_{i}.png")
    for i in range(n_val):
        make_sample(val_img / f"sample_{i}.jpg", val_mask / f"sample_{i}.png")

    print(f"Dummy data written to {out_root}")


if __name__ == "__main__":
    main()
