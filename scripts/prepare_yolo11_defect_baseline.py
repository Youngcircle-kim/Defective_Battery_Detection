"""Prepare single-stage YOLOv11-Seg defect baseline dataset.

Classes:
  0: damaged
  1: pollution

Notes:
- Background is implicit in YOLO and is not included as a class.
- Swelling is excluded because the dataset has no positive swelling samples.
- Battery outline is excluded because this baseline evaluates visible exterior defects only.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import (
    list_image_files,
    load_config,
    normalize_polygon,
    write_yolo_seg_label,
)
from src.json_label_loader import parse_json_label


CLASSES = {
    "damaged": 0,
    "pollution": 1,
}


def find_label_path(labels_dir: Path, image_stem: str) -> Path | None:
    candidate = labels_dir / f"{image_stem}.json"
    if candidate.exists():
        return candidate

    matches = list(labels_dir.rglob(f"{image_stem}.json"))
    if matches:
        return matches[0]

    return None


def copy_or_link_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        return

    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "copy":
        try:
            shutil.copy2(src, dst)
        except Exception:
            shutil.copyfile(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def convert_split(
    cfg: dict,
    images_dir: Path,
    labels_dir: Path,
    out_root: Path,
    split_name: str,
    mode: str,
):
    img_w = cfg["image"]["width"]
    img_h = cfg["image"]["height"]
    schema = cfg["json_schema"]

    out_img_dir = out_root / "images" / split_name
    out_lbl_dir = out_root / "labels" / split_name

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_files(images_dir)

    print(f"\n[{split_name}] images: {len(image_paths):,}")

    n_no_label = 0
    n_empty = 0
    n_damaged = 0
    n_pollution = 0
    n_images_with_defect = 0

    for img_path in tqdm(image_paths, desc=f"prepare-{split_name}"):
        label_path = find_label_path(labels_dir, img_path.stem)

        items = []

        if label_path is None:
            n_no_label += 1

        else:
            try:
                parsed = parse_json_label(label_path, schema)
            except Exception as e:
                print(f"[warn] JSON parse failed: {label_path.name} | {e}")
                parsed = {
                    "battery_outline": [],
                    "damaged": [],
                    "pollution": [],
                }

            for poly_px in parsed["damaged"]:
                poly_norm = normalize_polygon(poly_px, img_w, img_h)
                items.append((CLASSES["damaged"], poly_norm))
                n_damaged += 1

            for poly_px in parsed["pollution"]:
                poly_norm = normalize_polygon(poly_px, img_w, img_h)
                items.append((CLASSES["pollution"], poly_norm))
                n_pollution += 1

            if items:
                n_images_with_defect += 1
            else:
                n_empty += 1

        dst_img = out_img_dir / img_path.name
        dst_lbl = out_lbl_dir / f"{img_path.stem}.txt"

        copy_or_link_image(img_path, dst_img, mode)
        write_yolo_seg_label(dst_lbl, items)

    print(
        f"[{split_name}] damaged instances={n_damaged:,}, "
        f"pollution instances={n_pollution:,}, "
        f"images_with_defect={n_images_with_defect:,}, "
        f"empty_labels={n_empty:,}, "
        f"no_label={n_no_label:,}"
    )


def write_data_yaml(out_root: Path) -> Path:
    yaml_path = out_root / "data.yaml"

    yaml_text = f"""path: {out_root.resolve()}
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: damaged
  1: pollution
"""

    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--out", type=str, default="data_yolo11_defect_baseline")
    parser.add_argument(
        "--mode",
        type=str,
        default="symlink",
        choices=["symlink", "copy"],
        help="symlink saves disk space. copy creates independent dataset.",
    )
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--clear", action="store_true")

    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]

    out_root = Path(args.out)

    if args.clear and out_root.exists():
        shutil.rmtree(out_root)

    convert_split(
        cfg=cfg,
        images_dir=Path(paths["raw_train_images"]),
        labels_dir=Path(paths["raw_train_labels"]),
        out_root=out_root,
        split_name="train",
        mode=args.mode,
    )

    convert_split(
        cfg=cfg,
        images_dir=Path(paths["raw_val_images"]),
        labels_dir=Path(paths["raw_val_labels"]),
        out_root=out_root,
        split_name="val",
        mode=args.mode,
    )

    if args.include_test:
        test_images = Path("data/Test/image_data/images")
        test_labels = Path("data/Test/label_data/labels")

        if test_images.exists() and test_labels.exists():
            convert_split(
                cfg=cfg,
                images_dir=test_images,
                labels_dir=test_labels,
                out_root=out_root,
                split_name="test",
                mode=args.mode,
            )
        else:
            print("[warn] Test image/label directory not found. Skipping test split.")

    yaml_path = write_data_yaml(out_root)
    print(f"\ndata.yaml created: {yaml_path}")
    print("Done.")


if __name__ == "__main__":
    main()