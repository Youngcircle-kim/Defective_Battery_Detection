"""
원본 통합 라벨 → Stage1 / Stage2 데이터셋으로 분리.

Stage1 (Model A):
  - battery_outline (cls 0)
  - damaged_large   (cls 1)   ← bbox max(w,h) >= threshold

Stage2 (full size, patch 생성 입력):
  - damaged_small   (cls 0)   ← bbox max(w,h) <  threshold
  - pollution       (cls 1)
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
    polygon_size_metric,
    read_yolo_seg_label,
    write_yolo_seg_label,
)


def split_one_split(
    cfg: dict,
    images_dir: Path,
    labels_dir: Path,
    stage1_root: Path,
    stage2_root: Path,
    split_name: str,    # 'train' | 'val'
):
    img_w = cfg["image"]["width"]
    img_h = cfg["image"]["height"]

    src = cfg["source_classes"]
    s1 = cfg["stage1"]["classes"]
    s2 = cfg["stage2"]["classes"]
    threshold = cfg["damage_split"]["threshold_pixels"]
    method = cfg["damage_split"]["method"]

    img_out_1 = stage1_root / "images" / split_name
    lbl_out_1 = stage1_root / "labels" / split_name
    img_out_2 = stage2_root / "images" / split_name
    lbl_out_2 = stage2_root / "labels" / split_name
    for d in [img_out_1, lbl_out_1, img_out_2, lbl_out_2]:
        d.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_files(images_dir)
    print(f"[{split_name}] {len(image_paths)} images")

    n_large, n_small, n_battery, n_pollution = 0, 0, 0, 0

    for img_path in tqdm(image_paths, desc=f"split-{split_name}"):
        lbl_path = labels_dir / (img_path.stem + ".txt")
        items = read_yolo_seg_label(lbl_path)

        s1_items = []
        s2_items = []
        for cls_id, poly_norm in items:
            poly_px = poly_norm.copy()
            poly_px[:, 0] *= img_w
            poly_px[:, 1] *= img_h

            if cls_id == src["battery_outline"]:
                s1_items.append((s1["battery_outline"], poly_norm))
                n_battery += 1
            elif cls_id == src["damaged"]:
                size = polygon_size_metric(poly_px, method=method)
                if size >= threshold:
                    s1_items.append((s1["damaged_large"], poly_norm))
                    n_large += 1
                else:
                    s2_items.append((s2["damaged_small"], poly_norm))
                    n_small += 1
            elif cls_id == src["pollution"]:
                s2_items.append((s2["pollution"], poly_norm))
                n_pollution += 1

        # Stage1: 모든 이미지 복사 (배경도 학습)
        # 단, 라벨이 비어있으면 이미지만 두고 빈 라벨 파일 생성
        dst_img1 = img_out_1 / img_path.name
        if not dst_img1.exists():
            try:
                shutil.copy2(img_path, dst_img1)
            except Exception:
                shutil.copyfile(img_path, dst_img1)
        write_yolo_seg_label(lbl_out_1 / (img_path.stem + ".txt"), s1_items)

        # Stage2 full: patch 생성 시 사용. 모든 이미지+라벨 보관.
        dst_img2 = img_out_2 / img_path.name
        if not dst_img2.exists():
            try:
                shutil.copy2(img_path, dst_img2)
            except Exception:
                shutil.copyfile(img_path, dst_img2)
        write_yolo_seg_label(lbl_out_2 / (img_path.stem + ".txt"), s2_items)

    print(
        f"[{split_name}] battery={n_battery}, "
        f"damaged_large={n_large}, damaged_small={n_small}, pollution={n_pollution}"
    )


def write_data_yaml(stage1_root: Path, classes: dict, name: str = "data.yaml") -> Path:
    """Stage1용 YOLO data.yaml 생성."""
    inv = sorted(classes.items(), key=lambda kv: kv[1])
    names = [k for k, _ in inv]
    yaml_path = stage1_root / name
    yaml_text = (
        f"path: {stage1_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = cfg["paths"]
    stage1_root = Path(p["stage1_root"])
    stage2_root = Path(p["stage2_full_root"])

    # train
    split_one_split(
        cfg,
        Path(p["raw_train_images"]), Path(p["raw_train_labels"]),
        stage1_root, stage2_root, "train",
    )
    # val
    split_one_split(
        cfg,
        Path(p["raw_val_images"]), Path(p["raw_val_labels"]),
        stage1_root, stage2_root, "val",
    )

    # data.yaml for Stage1
    s1_yaml = write_data_yaml(stage1_root, cfg["stage1"]["classes"])
    print(f"[Stage1] data.yaml → {s1_yaml}")

    # Stage2 full에는 yaml 안 만듦 (patch 생성 후 patch root에서 만듦)
    print("Done. Next: scripts/02_generate_patches.py")


if __name__ == "__main__":
    main()
