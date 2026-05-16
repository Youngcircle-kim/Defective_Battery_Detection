"""Prepare YOLOv11-Seg sliding-crop baseline dataset in parallel.

Classes:
  0: battery_outline
  1: damaged
  2: pollution

Parallelization:
  - Image-level multiprocessing
  - Each worker loads one full image, generates sliding crops, clips labels, saves patches
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
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
    "battery_outline": 0,
    "damaged": 1,
    "pollution": 2,
}

_CFG = None


def _init_worker(cfg: dict):
    global _CFG
    _CFG = cfg
    cv2.setNumThreads(0)


def find_label_path(labels_dir: Path, image_stem: str) -> Path | None:
    candidate = labels_dir / f"{image_stem}.json"
    if candidate.exists():
        return candidate

    matches = list(labels_dir.rglob(f"{image_stem}.json"))
    if matches:
        return matches[0]

    return None


def ensure_xy_array(poly) -> np.ndarray:
    arr = np.asarray(poly, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return arr


def polygon_area(poly: np.ndarray) -> float:
    if poly is None or len(poly) < 3:
        return 0.0
    return float(abs(cv2.contourArea(poly.astype(np.float32))))


def clip_polygon_to_rect(
    poly: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> list[np.ndarray]:
    poly = ensure_xy_array(poly)

    if len(poly) < 3:
        return []

    rect = np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )

    try:
        area, inter = cv2.intersectConvexConvex(poly.astype(np.float32), rect)
        if area <= 1e-6 or inter is None:
            return []

        inter = inter.reshape(-1, 2)
        if len(inter) < 3:
            return []

        return [inter.astype(np.float32)]

    except Exception:
        inside = poly[
            (poly[:, 0] >= x1)
            & (poly[:, 0] <= x2)
            & (poly[:, 1] >= y1)
            & (poly[:, 1] <= y2)
        ]

        if len(inside) < 3:
            return []

        return [inside.astype(np.float32)]


def polygon_to_patch_local(poly_px: np.ndarray, crop_x: int, crop_y: int) -> np.ndarray:
    local = poly_px.copy().astype(np.float32)
    local[:, 0] -= crop_x
    local[:, 1] -= crop_y
    return local


def generate_sliding_positions(
    width: int,
    height: int,
    crop_size: int,
    stride: int,
) -> list[tuple[int, int]]:
    xs = list(range(0, max(1, width - crop_size + 1), stride))
    ys = list(range(0, max(1, height - crop_size + 1), stride))

    if not xs or xs[-1] != width - crop_size:
        xs.append(max(0, width - crop_size))

    if not ys or ys[-1] != height - crop_size:
        ys.append(max(0, height - crop_size))

    return [(x, y) for y in ys for x in xs]


def crop_with_padding(img: np.ndarray, x: int, y: int, crop_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    crop = img[y:min(y + crop_size, h), x:min(x + crop_size, w)]

    ch, cw = crop.shape[:2]

    if ch == crop_size and cw == crop_size:
        return crop

    padded = np.zeros((crop_size, crop_size, 3), dtype=img.dtype)
    padded[:ch, :cw] = crop
    return padded


def process_one_image(task):
    (
        img_path_str,
        label_path_str,
        out_img_dir_str,
        out_lbl_dir_str,
        crop_size,
        stride,
        min_polygon_area_px,
        negative_ratio,
        jpeg_quality,
        seed,
    ) = task

    cfg = _CFG
    schema = cfg["json_schema"]

    img_path = Path(img_path_str)
    label_path = Path(label_path_str) if label_path_str else None
    out_img_dir = Path(out_img_dir_str)
    out_lbl_dir = Path(out_lbl_dir_str)

    img = cv2.imread(str(img_path))
    if img is None:
        return {
            "ok": False,
            "image": img_path.name,
            "pos": 0,
            "neg": 0,
            "error": "image read failed",
        }

    h, w = img.shape[:2]

    all_polys: list[tuple[int, np.ndarray]] = []

    if label_path is not None and label_path.exists():
        try:
            parsed = parse_json_label(label_path, schema)
        except Exception as e:
            return {
                "ok": False,
                "image": img_path.name,
                "pos": 0,
                "neg": 0,
                "error": f"json parse failed: {e}",
            }

        for poly_px in parsed["battery_outline"]:
            all_polys.append((CLASSES["battery_outline"], ensure_xy_array(poly_px)))

        for poly_px in parsed["damaged"]:
            all_polys.append((CLASSES["damaged"], ensure_xy_array(poly_px)))

        for poly_px in parsed["pollution"]:
            all_polys.append((CLASSES["pollution"], ensure_xy_array(poly_px)))

    positions = generate_sliding_positions(w, h, crop_size, stride)

    positive_entries = []
    negative_entries = []

    for crop_x, crop_y in positions:
        crop_x2 = crop_x + crop_size
        crop_y2 = crop_y + crop_size

        labels = []

        for cls_id, poly_px in all_polys:
            clipped_list = clip_polygon_to_rect(
                poly_px,
                crop_x,
                crop_y,
                crop_x2,
                crop_y2,
            )

            for clipped in clipped_list:
                local_poly = polygon_to_patch_local(clipped, crop_x, crop_y)

                if polygon_area(local_poly) < min_polygon_area_px:
                    continue

                local_poly[:, 0] = np.clip(local_poly[:, 0], 0, crop_size)
                local_poly[:, 1] = np.clip(local_poly[:, 1], 0, crop_size)

                if len(local_poly) < 3:
                    continue

                poly_norm = normalize_polygon(local_poly, crop_size, crop_size)
                labels.append((cls_id, poly_norm))

        entry = (crop_x, crop_y, labels)

        if labels:
            positive_entries.append(entry)
        else:
            negative_entries.append(entry)

    rng = random.Random(seed + (hash(img_path.stem) & 0xFFFFFFFF))

    if positive_entries:
        n_neg_keep = int(len(positive_entries) * negative_ratio)
        n_neg_keep = min(n_neg_keep, len(negative_entries))
        negative_keep = rng.sample(negative_entries, n_neg_keep) if n_neg_keep > 0 else []
    else:
        negative_keep = rng.sample(negative_entries, min(1, len(negative_entries))) if negative_entries else []

    selected_entries = [(e, "pos") for e in positive_entries] + [
        (e, "neg") for e in negative_keep
    ]

    jpg_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

    n_pos = 0
    n_neg = 0

    for (crop_x, crop_y, labels), kind in selected_entries:
        crop_img = crop_with_padding(img, crop_x, crop_y, crop_size)

        patch_stem = f"{img_path.stem}_x{crop_x:04d}_y{crop_y:04d}"
        out_img_path = out_img_dir / f"{patch_stem}.jpg"
        out_lbl_path = out_lbl_dir / f"{patch_stem}.txt"

        cv2.imwrite(str(out_img_path), crop_img, jpg_params)
        write_yolo_seg_label(out_lbl_path, labels)

        if kind == "pos":
            n_pos += 1
        else:
            n_neg += 1

    return {
        "ok": True,
        "image": img_path.name,
        "pos": n_pos,
        "neg": n_neg,
        "error": None,
    }


def convert_split_parallel(
    cfg: dict,
    images_dir: Path,
    labels_dir: Path,
    out_root: Path,
    split_name: str,
    crop_size: int,
    stride: int,
    min_polygon_area_px: float,
    negative_ratio: float,
    jpeg_quality: int,
    seed: int,
    limit: int,
    workers: int,
    chunksize: int,
):
    out_img_dir = out_root / "images" / split_name
    out_lbl_dir = out_root / "labels" / split_name

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_files(images_dir)

    if limit > 0:
        image_paths = image_paths[:limit]

    tasks = []
    no_label = 0

    for img_path in image_paths:
        label_path = find_label_path(labels_dir, img_path.stem)
        if label_path is None:
            no_label += 1
            label_path_str = ""
        else:
            label_path_str = str(label_path)

        tasks.append(
            (
                str(img_path),
                label_path_str,
                str(out_img_dir),
                str(out_lbl_dir),
                crop_size,
                stride,
                min_polygon_area_px,
                negative_ratio,
                jpeg_quality,
                seed,
            )
        )

    print(f"\n[{split_name}] source images: {len(image_paths):,}")
    print(f"[{split_name}] workers={workers}, chunksize={chunksize}")
    print(f"[{split_name}] crop_size={crop_size}, stride={stride}, negative_ratio={negative_ratio}")
    print(f"[{split_name}] no_label_images={no_label:,}")

    total_pos = 0
    total_neg = 0
    error_count = 0

    with mp.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(cfg,),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(process_one_image, tasks, chunksize=chunksize),
            total=len(tasks),
            desc=f"sliding-{split_name}",
        ):
            if result["ok"]:
                total_pos += result["pos"]
                total_neg += result["neg"]
            else:
                error_count += 1
                if error_count <= 10:
                    print(f"[warn] {result['image']} | {result['error']}")

    print(
        f"[{split_name}] positive_patches={total_pos:,}, "
        f"negative_patches={total_neg:,}, "
        f"errors={error_count:,}"
    )


def write_data_yaml(out_root: Path) -> Path:
    yaml_path = out_root / "data.yaml"

    yaml_text = f"""path: {out_root.resolve()}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: battery_outline
  1: damaged
  2: pollution
"""

    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--out", type=str, default="data_yolo11_sliding_baseline")

    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--stride", type=int, default=320)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--min-polygon-area-px", type=float, default=9.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--workers", type=int, default=max(1, min(16, mp.cpu_count() - 1)))
    parser.add_argument("--chunksize", type=int, default=8)

    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]

    out_root = Path(args.out)

    if args.clear and out_root.exists():
        shutil.rmtree(out_root)

    convert_split_parallel(
        cfg=cfg,
        images_dir=Path(paths["raw_train_images"]),
        labels_dir=Path(paths["raw_train_labels"]),
        out_root=out_root,
        split_name="train",
        crop_size=args.crop_size,
        stride=args.stride,
        min_polygon_area_px=args.min_polygon_area_px,
        negative_ratio=args.negative_ratio,
        jpeg_quality=args.jpeg_quality,
        seed=args.seed,
        limit=args.limit,
        workers=args.workers,
        chunksize=args.chunksize,
    )

    convert_split_parallel(
        cfg=cfg,
        images_dir=Path(paths["raw_val_images"]),
        labels_dir=Path(paths["raw_val_labels"]),
        out_root=out_root,
        split_name="val",
        crop_size=args.crop_size,
        stride=args.stride,
        min_polygon_area_px=args.min_polygon_area_px,
        negative_ratio=args.negative_ratio,
        jpeg_quality=args.jpeg_quality,
        seed=args.seed + 1,
        limit=args.limit,
        workers=args.workers,
        chunksize=args.chunksize,
    )

    if args.include_test:
        test_images = Path("data/Test/image_data/images")
        test_labels = Path("data/Test/label_data/labels")

        if test_images.exists() and test_labels.exists():
            convert_split_parallel(
                cfg=cfg,
                images_dir=test_images,
                labels_dir=test_labels,
                out_root=out_root,
                split_name="test",
                crop_size=args.crop_size,
                stride=args.stride,
                min_polygon_area_px=args.min_polygon_area_px,
                negative_ratio=args.negative_ratio,
                jpeg_quality=args.jpeg_quality,
                seed=args.seed + 2,
                limit=args.limit,
                workers=args.workers,
                chunksize=args.chunksize,
            )
        else:
            print("[warn] Test image/label directory not found. Skipping test split.")

    yaml_path = write_data_yaml(out_root)

    print(f"\ndata.yaml created: {yaml_path}")
    print("Done.")


if __name__ == "__main__":
    main()