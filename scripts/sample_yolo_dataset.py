"""Create a sampled YOLO segmentation dataset.

Expected input structure:

data_stage1/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml

Output:

data_stage1_sample/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(image_dir: Path) -> list[Path]:
    images = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(images)


def has_label(image_path: Path, src_images_dir: Path, src_labels_dir: Path) -> bool:
    rel = image_path.relative_to(src_images_dir)
    label_path = src_labels_dir / rel.with_suffix(".txt")
    return label_path.exists()


def make_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def sample_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    n_samples: int | None,
    ratio: float | None,
    seed: int,
    mode: str,
) -> int:
    src_images_dir = src_root / "images" / split
    src_labels_dir = src_root / "labels" / split

    dst_images_dir = dst_root / "images" / split
    dst_labels_dir = dst_root / "labels" / split

    if not src_images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {src_images_dir}")
    if not src_labels_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {src_labels_dir}")

    images = collect_images(src_images_dir)

    # 라벨이 있는 이미지만 사용
    images = [
        img for img in images
        if has_label(img, src_images_dir, src_labels_dir)
    ]

    if not images:
        raise RuntimeError(f"No valid image-label pairs found in {split}")

    rng = random.Random(seed)

    if ratio is not None:
        k = max(1, int(len(images) * ratio))
    elif n_samples is not None:
        k = min(n_samples, len(images))
    else:
        k = len(images)

    sampled = rng.sample(images, k)

    for img_path in sampled:
        rel = img_path.relative_to(src_images_dir)
        label_path = src_labels_dir / rel.with_suffix(".txt")

        dst_img = dst_images_dir / rel
        dst_label = dst_labels_dir / rel.with_suffix(".txt")

        make_link_or_copy(img_path, dst_img, mode)
        make_link_or_copy(label_path, dst_label, mode)

    print(f"[{split}] sampled {len(sampled)} / {len(images)}")
    return len(sampled)


def write_data_yaml(dst_root: Path) -> None:
    data_yaml = dst_root / "data.yaml"
    content = f"""path: {dst_root.resolve()}
train: images/train
val: images/val

nc: 2
names:
  0: battery_outline
  1: damaged_large
"""
    data_yaml.write_text(content, encoding="utf-8")
    print(f"Created: {data_yaml}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="data_stage1")
    parser.add_argument("--dst", type=str, default="data_stage1_sample")

    parser.add_argument("--train-n", type=int, default=20000)
    parser.add_argument("--val-n", type=int, default=3000)

    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        type=str,
        default="symlink",
        choices=["symlink", "copy"],
        help="symlink saves disk space; copy creates independent files.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove destination directory before sampling.",
    )

    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if args.clear and dst_root.exists():
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)

    sample_split(
        src_root=src_root,
        dst_root=dst_root,
        split="train",
        n_samples=args.train_n,
        ratio=args.train_ratio,
        seed=args.seed,
        mode=args.mode,
    )

    sample_split(
        src_root=src_root,
        dst_root=dst_root,
        split="val",
        n_samples=args.val_n,
        ratio=args.val_ratio,
        seed=args.seed + 1,
        mode=args.mode,
    )

    write_data_yaml(dst_root)

    print("Sampling complete.")


if __name__ == "__main__":
    main()