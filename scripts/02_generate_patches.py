"""
Stage2 학습용 patch 데이터셋 생성 (병렬 버전).

핵심 차이:
  - multiprocessing.Pool으로 이미지 단위 병렬 처리
  - 워커 수 = CPU 코어 수 - 1 (또는 --workers로 지정)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path
from typing import Tuple

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import (
    list_image_files,
    load_config,
    normalize_polygon,
    read_yolo_seg_label,
)
from src.patch_utils import (
    clip_polygon_to_box,
    compute_roi_bbox,
    crop_patch,
    generate_patch_grid,
)


# 워커들이 공유할 전역 (fork로 inherit)
_CFG = None


def _init_worker(cfg_dict):
    """워커 프로세스 초기화 — config을 전역에 저장."""
    global _CFG
    _CFG = cfg_dict
    # OpenCV 스레드 비활성화 (워커 안에서 멀티스레드 충돌 방지)
    cv2.setNumThreads(0)


def _process_one_image(args) -> Tuple[int, int]:
    """
    단일 이미지에 대해 patch 생성 + 저장.

    Returns:
        (n_positive, n_negative)
    """
    img_path_str, s1_lbl_path_str, s2_full_lbl_path_str, \
        out_img_dir_str, out_lbl_dir_str = args

    cfg = _CFG
    img_w = cfg["image"]["width"]
    img_h = cfg["image"]["height"]
    margin = cfg["inference"]["roi_margin"]
    patch_size = cfg["stage2"]["patch_size"]
    stride = cfg["stage2"]["patch_stride"]
    neg_ratio = cfg["stage2"]["negative_patch_ratio"]
    min_area = cfg["stage2"]["min_polygon_area_px"]
    BATTERY_CLS = cfg["stage1"]["classes"]["battery_outline"]

    img_path = Path(img_path_str)
    s1_lbl_path = Path(s1_lbl_path_str)
    s2_full_lbl_path = Path(s2_full_lbl_path_str)
    out_img = Path(out_img_dir_str)
    out_lbl = Path(out_lbl_dir_str)

    # ── 1. battery_outline polygon 가져오기 (Stage1 라벨에서)
    s1_items = read_yolo_seg_label(s1_lbl_path)
    battery_polys_px = []
    for cls, p in s1_items:
        if cls == BATTERY_CLS:
            pp = p.copy()
            pp[:, 0] *= img_w
            pp[:, 1] *= img_h
            battery_polys_px.append(pp)

    if battery_polys_px:
        x1, y1, x2, y2 = compute_roi_bbox(battery_polys_px, img_w, img_h, margin=margin)
    else:
        x1, y1, x2, y2 = 0, 0, img_w, img_h

    # ── 2. defect polygons 가져오기 (Stage2 full 라벨에서)
    s2_items = read_yolo_seg_label(s2_full_lbl_path)
    defect_polys_px = []
    for cls, p in s2_items:
        pp = p.copy()
        pp[:, 0] *= img_w
        pp[:, 1] *= img_h
        defect_polys_px.append((cls, pp))

    # ── 3. 이미지 로드 + ROI crop
    img = cv2.imread(str(img_path))
    if img is None:
        return (0, 0)
    roi = img[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]

    # ── 4. patch grid + 각 patch 처리
    positions = generate_patch_grid(roi_w, roi_h, patch_size, stride)

    positive_patches = []
    negative_patches = []

    for (px, py) in positions:
        px2 = px + patch_size
        py2 = py + patch_size

        patch_labels = []
        for cls, poly_px in defect_polys_px:
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

    # ── 5. negative sampling
    rng = random.Random(hash(img_path.stem) & 0xFFFFFFFF)
    if negative_patches and positive_patches:
        n_keep = int(len(positive_patches) * neg_ratio)
        n_keep = min(n_keep, len(negative_patches))
        negative_keep = rng.sample(negative_patches, n_keep)
    elif positive_patches:
        negative_keep = []
    else:
        n_keep = min(1, len(negative_patches))
        negative_keep = rng.sample(negative_patches, n_keep) if n_keep > 0 else []

    # ── 6. 저장 (jpg 품질 90으로 조정 — 92 → 90)
    n_pos = 0
    n_neg = 0
    all_patches = [(p, "pos") for p in positive_patches] + [
        (p, "neg") for p in negative_keep
    ]
    jpg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    for entry, kind in all_patches:
        patch_img, labels, (px, py) = entry
        stem = f"{img_path.stem}_p{px:04d}_{py:04d}"
        cv2.imwrite(str(out_img / f"{stem}.jpg"), patch_img, jpg_params)

        # YOLO seg 라벨 저장
        label_path = out_lbl / f"{stem}.txt"
        if labels:
            lines = []
            for cls, polygon in labels:
                if polygon.shape[0] < 3:
                    continue
                flat = polygon.reshape(-1).tolist()
                coord_str = " ".join(f"{v:.6f}" for v in flat)
                lines.append(f"{int(cls)} {coord_str}")
            label_path.write_text("\n".join(lines), encoding="utf-8")
        else:
            label_path.touch()  # 빈 라벨 (negative)

        if kind == "pos":
            n_pos += 1
        else:
            n_neg += 1

    return (n_pos, n_neg)


def process_split(cfg: dict, split_name: str, num_workers: int):
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
    print(f"[{split_name}] {len(image_paths)} source images, workers={num_workers}")

    # 워커에 전달할 인자 리스트
    tasks = []
    for img_path in image_paths:
        s1_lbl = s1_lbl_dir / (img_path.stem + ".txt")
        s2_lbl = s2_full_lbl_dir / (img_path.stem + ".txt")
        tasks.append((
            str(img_path),
            str(s1_lbl),
            str(s2_lbl),
            str(out_img),
            str(out_lbl),
        ))

    n_pos_total = 0
    n_neg_total = 0

    # 병렬 처리
    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(cfg,),
    ) as pool:
        with tqdm(total=len(tasks), desc=f"patches-{split_name}") as pbar:
            for n_pos, n_neg in pool.imap_unordered(
                _process_one_image, tasks, chunksize=8
            ):
                n_pos_total += n_pos
                n_neg_total += n_neg
                pbar.update(1)
                pbar.set_postfix(pos=n_pos_total, neg=n_neg_total)

    print(f"[{split_name}] positive_patches={n_pos_total}, negative_patches={n_neg_total}")


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
    ap.add_argument("--workers", type=int, default=0,
                    help="0이면 CPU 코어 수 - 1 사용")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.workers <= 0:
        num_workers = max(1, mp.cpu_count() - 1)
    else:
        num_workers = args.workers
    print(f"▶ CPU cores: {mp.cpu_count()}, workers: {num_workers}")

    process_split(cfg, "train", num_workers)
    process_split(cfg, "val", num_workers)

    yaml_path = write_data_yaml(
        Path(cfg["paths"]["stage2_patches_root"]), cfg["stage2"]["classes"]
    )
    print(f"[Stage2] data.yaml → {yaml_path}")


if __name__ == "__main__":
    main()