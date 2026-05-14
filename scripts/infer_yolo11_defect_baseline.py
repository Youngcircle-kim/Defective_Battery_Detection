"""Inference for single-stage YOLOv11-Seg defect baseline.

Classes:
  0: damaged
  1: pollution
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from src.data_utils import list_image_files
from src.postprocess import mask_to_polygons


CLASS_NAMES = {
    0: "damaged",
    1: "pollution",
}

COLORS = {
    "damaged": (50, 100, 255),
    "pollution": (50, 220, 50),
}


def overlay_mask(image: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45):
    out = image.copy()
    color_layer = np.zeros_like(image)
    color_layer[mask] = color
    out = cv2.addWeighted(out, 1.0, color_layer, alpha, 0)
    return out


def draw_legend(image: np.ndarray):
    out = image.copy()
    x, y = 20, 30

    for cls_name, color in COLORS.items():
        cv2.rectangle(out, (x, y - 15), (x + 20, y + 5), color, -1)
        cv2.putText(
            out,
            cls_name,
            (x + 30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        y += 35

    return out


def predict_one(model: YOLO, img: np.ndarray, imgsz: int, conf: float, iou: float, device: str):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="runs/baseline_inference")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="0")

    args = parser.parse_args()

    weights = Path(args.weights)
    input_path = Path(args.image)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    model = YOLO(str(weights))

    if input_path.is_dir():
        image_paths = list_image_files(input_path)
    else:
        image_paths = [input_path]

    if not image_paths:
        raise RuntimeError(f"No images found: {input_path}")

    for img_path in tqdm(image_paths, desc="baseline inference"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warn] read failed: {img_path}")
            continue

        masks = predict_one(
            model=model,
            img=img,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
        )

        vis = img.copy()
        for cls_name, mask in masks.items():
            vis = overlay_mask(vis, mask, COLORS[cls_name])

        vis = draw_legend(vis)

        cv2.imwrite(str(output_dir / f"{img_path.stem}_baseline_vis.png"), vis)

        mask_dir = output_dir / f"{img_path.stem}_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

        for cls_name, mask in masks.items():
            cv2.imwrite(
                str(mask_dir / f"{cls_name}.png"),
                (mask.astype(np.uint8) * 255),
            )

        polygons = {cls: mask_to_polygons(m) for cls, m in masks.items()}
        with open(output_dir / f"{img_path.stem}_polygons.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image": img_path.name,
                    "image_size": {
                        "width": img.shape[1],
                        "height": img.shape[0],
                    },
                    "polygons": polygons,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()