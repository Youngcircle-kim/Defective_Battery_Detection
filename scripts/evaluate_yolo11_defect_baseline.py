"""Evaluate single-stage YOLOv11-Seg defect baseline.

GT:
  JSON defects only
  - Damaged
  - Pollution

Pred:
  YOLO classes
  - damaged
  - pollution
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import list_image_files, load_config
from src.json_label_loader import parse_json_label
from src.metrics import aggregate_metrics, evaluate_sample


CLASS_NAMES = {
    0: "damaged",
    1: "pollution",
}

COLORS = {
    "damaged": (50, 100, 255),
    "pollution": (50, 220, 50),
}


def find_label_path(label_dir: Path, img_path: Path) -> Path | None:
    candidate = label_dir / f"{img_path.stem}.json"
    if candidate.exists():
        return candidate

    matches = list(label_dir.rglob(f"{img_path.stem}.json"))
    if matches:
        return matches[0]

    return None


def gt_masks_from_label(
    label_path: Path | None,
    w: int,
    h: int,
    schema: dict,
) -> Dict[str, np.ndarray]:
    masks = {
        "damaged": np.zeros((h, w), dtype=bool),
        "pollution": np.zeros((h, w), dtype=bool),
    }

    if label_path is None or not label_path.exists():
        return masks

    try:
        parsed = parse_json_label(label_path, schema)
    except Exception as e:
        print(f"[warn] JSON parse failed: {label_path} | {e}")
        return masks

    for cls_name in ["damaged", "pollution"]:
        for poly_px in parsed[cls_name]:
            if poly_px is None or len(poly_px) < 3:
                continue

            poly_int = poly_px.astype(np.int32)
            layer = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(layer, [poly_int], 1)
            masks[cls_name] |= layer.astype(bool)

    return masks


def predict_one(
    model: YOLO,
    img: np.ndarray,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
) -> Dict[str, np.ndarray]:
    h, w = img.shape[:2]

    masks_by_class = {
        "damaged": np.zeros((h, w), dtype=bool),
        "pollution": np.zeros((h, w), dtype=bool),
    }

    results = model.predict(
        source=img,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )

    r = results[0]

    if r.masks is None or r.boxes is None:
        return masks_by_class

    masks = r.masks.data.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)

    for mask, cls_id in zip(masks, classes):
        cls_name = CLASS_NAMES.get(cls_id)
        if cls_name is None:
            continue

        mask_resized = cv2.resize(
            mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        masks_by_class[cls_name] |= mask_resized > 0.5

    return masks_by_class


def overlay_mask(image: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45):
    out = image.copy()
    color_layer = np.zeros_like(image)
    color_layer[mask] = color
    out = cv2.addWeighted(out, 1.0, color_layer, alpha, 0)
    return out


def make_vis(img: np.ndarray, masks: Dict[str, np.ndarray]):
    vis = img.copy()
    for cls_name, mask in masks.items():
        vis = overlay_mask(vis, mask, COLORS[cls_name])
    return vis


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--weights", type=str, required=True)

    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--label-dir", type=str, default=None)

    parser.add_argument("--output", type=str, default="runs/baseline_eval")
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save-vis", action="store_true")

    args = parser.parse_args()

    cfg = load_config(args.config)
    schema = cfg["json_schema"]

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    if args.image_dir is not None:
        img_dir = Path(args.image_dir)
        if args.label_dir is None:
            raise ValueError("--image-dir 사용 시 --label-dir도 필요합니다.")
        lbl_dir = Path(args.label_dir)

    elif args.val_dir is not None:
        root = Path(args.val_dir)
        img_dir = root / "image_data" / "images"
        lbl_dir = root / "label_data" / "labels"

    else:
        img_dir = Path(cfg["paths"]["raw_val_images"])
        lbl_dir = Path(cfg["paths"]["raw_val_labels"])

    if not img_dir.exists():
        raise FileNotFoundError(f"image dir not found: {img_dir}")
    if not lbl_dir.exists():
        raise FileNotFoundError(f"label dir not found: {lbl_dir}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    vis_dir = out / "visualizations"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    print("[Evaluation Config]")
    print(f"  image_dir : {img_dir}")
    print(f"  label_dir : {lbl_dir}")
    print(f"  weights   : {weights}")
    print(f"  imgsz     : {args.imgsz}")
    print(f"  conf      : {args.conf}")
    print(f"  iou       : {args.iou}")

    model = YOLO(str(weights))

    img_paths = list_image_files(img_dir)
    if args.limit > 0:
        img_paths = img_paths[: args.limit]

    print(f"평가 대상: {len(img_paths)}장")

    sample_metrics = []
    per_image_log = []
    missing_labels = 0

    for img_path in tqdm(img_paths, desc="evaluate baseline"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warn] read failed: {img_path}")
            continue

        h, w = img.shape[:2]

        label_path = find_label_path(lbl_dir, img_path)
        if label_path is None:
            missing_labels += 1

        gt = gt_masks_from_label(label_path, w, h, schema)
        pred = predict_one(
            model=model,
            img=img,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
        )

        m = evaluate_sample(pred, gt)
        sample_metrics.append(m)

        per_image_log.append(
            {
                "image": img_path.name,
                "label": None if label_path is None else str(label_path),
                "metrics": m,
            }
        )

        if args.save_vis:
            gt_vis = make_vis(img, gt)
            pred_vis = make_vis(img, pred)

            compare = np.concatenate([gt_vis, pred_vis], axis=1)
            cv2.putText(
                compare,
                "GT",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3,
            )
            cv2.putText(
                compare,
                "Prediction",
                (gt_vis.shape[1] + 30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3,
            )
            cv2.imwrite(str(vis_dir / f"{img_path.stem}_compare.png"), compare)

    if not sample_metrics:
        raise RuntimeError("No valid samples evaluated.")

    agg = aggregate_metrics(sample_metrics)

    print("\n========== Aggregated Metrics ==========")
    print(f"missing_labels: {missing_labels}")
    for cls, sub in agg.items():
        print(f"[{cls}]")
        for k, v in sub.items():
            print(f"   {k:>16}: {v:.4f}")

    with open(out / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "image_dir": str(img_dir),
                    "label_dir": str(lbl_dir),
                    "weights": str(weights),
                    "imgsz": args.imgsz,
                    "conf": args.conf,
                    "iou": args.iou,
                    "missing_labels": missing_labels,
                    "num_images": len(sample_metrics),
                },
                "metrics": agg,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(out / "metrics_per_image.json", "w", encoding="utf-8") as f:
        json.dump(per_image_log, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()