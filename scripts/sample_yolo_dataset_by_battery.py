"""Create a sampled YOLO segmentation dataset by battery ID.

This script samples images by battery/group ID to prevent data leakage.

Expected input:

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
import re
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def get_label_path(
    image_path: Path,
    src_images_dir: Path,
    src_labels_dir: Path,
) -> Path:
    rel = image_path.relative_to(src_images_dir)
    return src_labels_dir / rel.with_suffix(".txt")


def has_label(
    image_path: Path,
    src_images_dir: Path,
    src_labels_dir: Path,
) -> bool:
    return get_label_path(image_path, src_images_dir, src_labels_dir).exists()


def extract_battery_id(image_path: Path, id_regex: str | None = None) -> str:
    """
    Extract battery ID from filename.

    If id_regex is provided, the first capture group is used.
    Otherwise, a conservative default is used:
    - remove extension
    - split by '_' or '-'
    - use the first token as battery_id

    Example:
        BAT001_0001.jpg       -> BAT001
        BAT001-view-01.jpg    -> BAT001
        B12345_left_001.jpg   -> B12345
    """

    stem = image_path.stem

    if id_regex:
        match = re.search(id_regex, stem)
        if not match:
            raise ValueError(
                f"Cannot extract battery_id from {image_path.name} "
                f"using regex: {id_regex}"
            )
        return match.group(1)

    # default rule
    tokens = re.split(r"[_\-]", stem)
    return tokens[0]


def group_images_by_battery_id(
    images: list[Path],
    id_regex: str | None,
) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)

    for img_path in images:
        battery_id = extract_battery_id(img_path, id_regex=id_regex)
        groups[battery_id].append(img_path)

    return dict(groups)


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


def select_groups_by_image_count(
    groups: dict[str, list[Path]],
    target_image_count: int | None,
    target_group_count: int | None,
    ratio: float | None,
    seed: int,
) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    battery_ids = list(groups.keys())
    rng.shuffle(battery_ids)

    if ratio is not None:
        k_groups = max(1, int(len(battery_ids) * ratio))
        selected_ids = battery_ids[:k_groups]

    elif target_group_count is not None:
        k_groups = min(target_group_count, len(battery_ids))
        selected_ids = battery_ids[:k_groups]

    elif target_image_count is not None:
        selected_ids = []
        total_images = 0

        for battery_id in battery_ids:
            selected_ids.append(battery_id)
            total_images += len(groups[battery_id])

            if total_images >= target_image_count:
                break

    else:
        selected_ids = battery_ids

    return {battery_id: groups[battery_id] for battery_id in selected_ids}


def sample_split_by_battery(
    src_root: Path,
    dst_root: Path,
    split: str,
    target_image_count: int | None,
    target_group_count: int | None,
    ratio: float | None,
    seed: int,
    mode: str,
    id_regex: str | None,
) -> tuple[int, int]:
    src_images_dir = src_root / "images" / split
    src_labels_dir = src_root / "labels" / split

    dst_images_dir = dst_root / "images" / split
    dst_labels_dir = dst_root / "labels" / split

    if not src_images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {src_images_dir}")
    if not src_labels_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {src_labels_dir}")

    images = collect_images(src_images_dir)

    images = [
        img for img in images
        if has_label(img, src_images_dir, src_labels_dir)
    ]

    if not images:
        raise RuntimeError(f"No valid image-label pairs found in {split}")

    groups = group_images_by_battery_id(images, id_regex=id_regex)

    selected_groups = select_groups_by_image_count(
        groups=groups,
        target_image_count=target_image_count,
        target_group_count=target_group_count,
        ratio=ratio,
        seed=seed,
    )

    copied_image_count = 0

    for battery_id, group_images in selected_groups.items():
        for img_path in group_images:
            rel = img_path.relative_to(src_images_dir)
            label_path = src_labels_dir / rel.with_suffix(".txt")

            dst_img = dst_images_dir / rel
            dst_label = dst_labels_dir / rel.with_suffix(".txt")

            make_link_or_copy(img_path, dst_img, mode)
            make_link_or_copy(label_path, dst_label, mode)

            copied_image_count += 1

    print(
        f"[{split}] selected {len(selected_groups)} battery IDs "
        f"/ {len(groups)} total IDs"
    )
    print(
        f"[{split}] selected {copied_image_count} images "
        f"/ {len(images)} total images"
    )

    return len(selected_groups), copied_image_count


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
    parser.add_argument("--dst", type=str, default="data_stage1_sample_by_battery")

    parser.add_argument("--train-n", type=int, default=20000)
    parser.add_argument("--val-n", type=int, default=3000)

    parser.add_argument("--train-groups", type=int, default=None)
    parser.add_argument("--val-groups", type=int, default=None)

    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--mode",
        type=str,
        default="symlink",
        choices=["symlink", "copy"],
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove destination directory before sampling.",
    )

    parser.add_argument(
        "--id-regex",
        type=str,
        default=None,
        help=(
            "Regex for extracting battery ID from filename. "
            "Use the first capture group as battery ID. "
            "Example: '^([^_]+)_' extracts token before first underscore."
        ),
    )

    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if args.clear and dst_root.exists():
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)

    sample_split_by_battery(
        src_root=src_root,
        dst_root=dst_root,
        split="train",
        target_image_count=args.train_n,
        target_group_count=args.train_groups,
        ratio=args.train_ratio,
        seed=args.seed,
        mode=args.mode,
        id_regex=args.id_regex,
    )

    sample_split_by_battery(
        src_root=src_root,
        dst_root=dst_root,
        split="val",
        target_image_count=args.val_n,
        target_group_count=args.val_groups,
        ratio=args.val_ratio,
        seed=args.seed + 1,
        mode=args.mode,
        id_regex=args.id_regex,
    )

    write_data_yaml(dst_root)

    print("Battery-ID based sampling complete.")


if __name__ == "__main__":
    main()