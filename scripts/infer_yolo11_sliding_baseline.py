"""Sliding-window inference for YOLOv11-Seg baseline.

Classes:
  0: battery_outline
  1: damaged
  2: pollution

Process:
  - Full image -> 640 sliding crops
  - YOLO prediction per crop
  - Restore crop masks to original coordinates
  - Merge class masks on full image
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
    0: "battery_outline",
    1: "damaged",
    2: "pollution",
}

COLORS = {
    "battery_outline": (255, 100, 50),
    "damaged": (50, 100, 255),
    "pollution": (50, 220, 50),
}


def generate_sliding_positions(width: int, height: int, crop_size: int, stride: int) -> list[tuple[int, int]]:
    xs = list(range(0, max(1, width - crop_size + 1), stride))
    ys = list(range(0, max(1, height - crop_size + 1), stride))

    if not xs or xs[-1] != width - crop_size:
        xs.append(max(0, width - crop_size))

    if not ys or ys[-1] != height - crop_size:
        ys.append(max(0, height - crop_size))

    return [(x, y) for y in ys for x in xs]


def crop_with_padding(img: np.ndarray, x: int, y: int, crop_size: int) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    crop = img[y:min(y + crop_size, h), x:min(x + crop_size, w)]
    ch, cw = crop.shape[:2]

    if ch == crop_size and cw == crop_size:
        return crop, cw, ch

    padded = np.zeros((crop_size, crop_size, 3), dtype=img.dtype)
    padded[:ch, :cw] = crop
    return padded, cw, ch


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


def predict_sliding(
    model: YOLO,
    img: np.ndarray,
    crop_size: int,
    stride: int,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
) -> dict[str, np.ndarray]:
    h, w = img.shape[:2]

    masks_by_class = {
        "battery_outline": np.zeros((h, w), dtype=bool),
        "damaged": np.zeros((h, w), dtype=bool),
        "pollution": np.zeros((h, w), dtype=bool),
    }

    positions = generate_sliding_positions(w, h, crop_size, stride)

    for crop_x, crop_y in positions:
        crop, valid_w, valid_h = crop_with_padding(img, crop_x, crop_y, crop_size)

        results = model.predict(
            source=crop,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )

        r = results[0]

        if r.masks is None or r.boxes is None:
            continue

        pred_masks = r.masks.data.cpu().numpy()
        pred_classes = r.boxes.cls.cpu().numpy().astype(int)

        for mask, cls_id in zip(pred_masks, pred_classes):
            cls_name = CLASS_NAMES.get(cls_id)
            if cls_name is None:
                continue

            mask_resized = cv2.resize(
                mask,
                (crop_size, crop_size),
                interpolation=cv2.INTER_NEAREST,
            ) > 0.5

            # padding 영역 제거
            mask_valid = mask_resized[:valid_h, :valid_w]

            y2 = min(crop_y + valid_h, h)
            x2 = min(crop_x + valid_w, w)

            masks_by_class[cls_name][crop_y:y2, crop_x:x2] |= mask_valid[: y2 - crop_y, : x2 - crop_x]

    return masks_by_class


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="runs/sliding_inference")

    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--stride", type=int, default=320)
    parser.add_argument("--imgsz", type=int, default=640)

    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="0")

    args = parser.parse_args()

    weights = Path(args.weights)
    input_path = Path(args.image)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    model = YOLO(str(weights))

    if input_path.is_dir():
        image_paths = list_image_files(input_path)
    else:
        image_paths = [input_path]

    for img_path in tqdm(image_paths, desc="sliding inference"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warn] read failed: {img_path}")
            continue

        masks = predict_sliding(
            model=model,
            img=img,
            crop_size=args.crop_size,
            stride=args.stride,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
        )

        vis = img.copy()
        for cls_name, mask in masks.items():
            vis = overlay_mask(vis, mask, COLORS[cls_name])

        vis = draw_legend(vis)
        cv2.imwrite(str(out / f"{img_path.stem}_sliding_vis.png"), vis)

        mask_dir = out / f"{img_path.stem}_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

        for cls_name, mask in masks.items():
            cv2.imwrite(str(mask_dir / f"{cls_name}.png"), (mask.astype(np.uint8) * 255))

        polygons = {cls: mask_to_polygons(m) for cls, m in masks.items()}
        with open(out / f"{img_path.stem}_polygons.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image": img_path.name,
                    "image_size": {"width": img.shape[1], "height": img.shape[0]},
                    "polygons": polygons,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()