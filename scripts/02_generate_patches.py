"""
Stage2 학습용 patch 데이터셋 생성.

처리 순서:
  1. Stage2 full 라벨 (damaged_small + pollution) 로드
  2. Stage1 라벨에서 battery_outline 가져와 ROI bbox 계산 (10% margin)
  3. ROI 안에서 320x320 patch (stride 160) 생성
  4. 각 patch에 polygon 클리핑
  5. positive patch (defect 있는 것) 전부 + negative ratio만큼 sampling
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import (
    list_image_files,
    load_config,
    normalize_polygon,
    read_yolo_seg_label,
    write_yolo_seg_label,
)
from src.patch_utils import (
    clip_polygon_to_box,
    compute_roi_bbox,
    crop_patch,
    generate_patch_grid,
)


def process_split(cfg: dict, split_name: str):
    img_w = cfg["image"]["width"]
    img_h = cfg["image"]["height"]
    margin = cfg["inference"]["roi_margin"]
    patch_size = cfg["stage2"]["patch_size"]
    stride = cfg["stage2"]["patch_stride"]
    neg_ratio = cfg["stage2"]["negative_patch_ratio"]
    min_area = cfg["stage2"]["min_polygon_area_px"]

    s1_classes = cfg["stage1"]["classes"]
    BATTERY_CLS = s1_classes["battery_outline"]

    stage1_root = Path(cfg["paths"]["stage1_root"])
    stage2_full_root = Path(cfg["paths"]["stage2_full_root"])
    patches_root = Path(cfg["paths"]["stage2_patches_root"])

    s1_img_dir = stage1_root / "images" / split_name
    s1_lbl_dir = stage1_root / "labels" / split_name
    s2_full_lbl_dir = stage2_full_root / "labels" / split_name

    out_img = patches_root / "images" / split_name
    out_lbl = patches_root / "labels" / split_name
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_files(s1_img_dir)
    print(f"[{split_name}] {len(image_paths)} source images")

    n_pos, n_neg = 0, 0
    rng = random.Random(42)

    for img_path in tqdm(image_paths, desc=f"patches-{split_name}"):
        # battery_outline polygon (Stage1 label에서)
        s1_items = read_yolo_seg_label(s1_lbl_dir / (img_path.stem + ".txt"))
        battery_polys_px = []
        for cls, p in s1_items:
            if cls == BATTERY_CLS:
                pp = p.copy()
                pp[:, 0] *= img_w
                pp[:, 1] *= img_h
                battery_polys_px.append(pp)
        if not battery_polys_px:
            # 배터리가 없으면 이미지 전체를 ROI로
            roi_box = (0, 0, img_w, img_h)
        else:
            roi_box = compute_roi_bbox(battery_polys_px, img_w, img_h, margin=margin)

        # Stage2 polygons (defect들)
        s2_items = read_yolo_seg_label(s2_full_lbl_dir / (img_path.stem + ".txt"))
        defect_polys_px = []
        for cls, p in s2_items:
            pp = p.copy()
            pp[:, 0] *= img_w
            pp[:, 1] *= img_h
            defect_polys_px.append((cls, pp))

        # 이미지 로드
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        x1, y1, x2, y2 = roi_box
        roi = img[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        positions = generate_patch_grid(roi_w, roi_h, patch_size, stride)

        positive_patches = []
        negative_patches = []

        for (px, py) in positions:
            # patch 영역 (ROI 좌표계)
            px2 = px + patch_size
            py2 = py + patch_size

            # 각 defect polygon을 patch에 클리핑
            patch_labels = []
            for cls, poly_px in defect_polys_px:
                # 먼저 ROI 좌표계로 옮기기
                poly_in_roi = poly_px.copy()
                poly_in_roi[:, 0] -= x1
                poly_in_roi[:, 1] -= y1
                pieces = clip_polygon_to_box(
                    poly_in_roi, (px, py, px2, py2), min_area=min_area
                )
                for piece_local in pieces:
                    poly_norm = normalize_polygon(piece_local, patch_size, patch_size)
                    patch_labels.append((cls, poly_norm))

            patch_img = crop_patch(roi, px, py, patch_size)
            entry = (patch_img, patch_labels, (px, py))
            if patch_labels:
                positive_patches.append(entry)
            else:
                negative_patches.append(entry)

        # negative sampling
        if negative_patches and positive_patches:
            n_keep = int(len(positive_patches) * neg_ratio)
            n_keep = min(n_keep, len(negative_patches))
            negative_keep = rng.sample(negative_patches, n_keep)
        elif positive_patches:
            negative_keep = []
        else:
            # positive 없으면 negative 적당히만 (전체 1개 정도)
            n_keep = min(1, len(negative_patches))
            negative_keep = rng.sample(negative_patches, n_keep) if n_keep > 0 else []

        # 저장
        all_patches = [(p, "pos") for p in positive_patches] + [
            (p, "neg") for p in negative_keep
        ]
        for entry, kind in all_patches:
            patch_img, labels, (px, py) = entry
            stem = f"{img_path.stem}_p{px:04d}_{py:04d}"
            cv2.imwrite(str(out_img / f"{stem}.jpg"), patch_img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            write_yolo_seg_label(out_lbl / f"{stem}.txt", labels)
            if kind == "pos":
                n_pos += 1
            else:
                n_neg += 1

    print(f"[{split_name}] positive_patches={n_pos}, negative_patches={n_neg}")


def write_data_yaml(patches_root: Path, classes: dict) -> Path:
    inv = sorted(classes.items(), key=lambda kv: kv[1])
    names = [k for k, _ in inv]
    yaml_path = patches_root / "data.yaml"
    txt = (
        f"path: {patches_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    yaml_path.write_text(txt, encoding="utf-8")
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    process_split(cfg, "train")
    process_split(cfg, "val")

    yaml_path = write_data_yaml(
        Path(cfg["paths"]["stage2_patches_root"]), cfg["stage2"]["classes"]
    )
    print(f"[Stage2] data.yaml → {yaml_path}")


if __name__ == "__main__":
    main()
