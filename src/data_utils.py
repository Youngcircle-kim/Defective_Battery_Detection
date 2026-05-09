"""YOLO segmentation 라벨 I/O 및 polygon 유틸."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml


def load_config(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_yolo_seg_label(label_path: str | os.PathLike) -> List[Tuple[int, np.ndarray]]:
    """
    YOLO segmentation 라벨 파일 읽기.
    Format (per line): class_id x1 y1 x2 y2 ... xn yn  (정규화된 [0,1])

    Returns:
        list of (class_id, polygon_normalized) where polygon_normalized
        is np.ndarray of shape (N, 2) with values in [0, 1].
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return []

    items: List[Tuple[int, np.ndarray]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class + 최소 3개 점 (6 좌표)
                continue
            cls = int(parts[0])
            coords = np.array(parts[1:], dtype=np.float32)
            if coords.size % 2 != 0:
                continue
            polygon = coords.reshape(-1, 2)
            items.append((cls, polygon))
    return items


def write_yolo_seg_label(
    label_path: str | os.PathLike,
    items: List[Tuple[int, np.ndarray]],
) -> None:
    """polygons를 YOLO seg 라벨로 저장. polygon은 정규화된 [0,1] 좌표."""
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cls, polygon in items:
        if polygon.shape[0] < 3:
            continue
        flat = polygon.reshape(-1).tolist()
        coord_str = " ".join(f"{v:.6f}" for v in flat)
        lines.append(f"{int(cls)} {coord_str}")
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def denormalize_polygon(poly_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    """normalized [0,1] → pixel coordinates."""
    out = poly_norm.copy()
    out[:, 0] *= w
    out[:, 1] *= h
    return out


def normalize_polygon(poly_px: np.ndarray, w: int, h: int) -> np.ndarray:
    """pixel → normalized [0,1]."""
    out = poly_px.astype(np.float32).copy()
    out[:, 0] /= w
    out[:, 1] /= h
    return np.clip(out, 0.0, 1.0)


def polygon_bbox(poly_px: np.ndarray) -> Tuple[float, float, float, float]:
    """polygon → (xmin, ymin, xmax, ymax) in same units as input."""
    xmin = float(poly_px[:, 0].min())
    ymin = float(poly_px[:, 1].min())
    xmax = float(poly_px[:, 0].max())
    ymax = float(poly_px[:, 1].max())
    return xmin, ymin, xmax, ymax


def polygon_size_metric(
    poly_px: np.ndarray, method: str = "bbox_max_side"
) -> float:
    """damage 분류용 크기 지표."""
    if method == "bbox_max_side":
        x1, y1, x2, y2 = polygon_bbox(poly_px)
        return max(x2 - x1, y2 - y1)
    elif method == "bbox_area_sqrt":
        x1, y1, x2, y2 = polygon_bbox(poly_px)
        return float(np.sqrt(max(0.0, (x2 - x1) * (y2 - y1))))
    elif method == "polygon_area_sqrt":
        # shoelace
        x = poly_px[:, 0]
        y = poly_px[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return float(np.sqrt(max(0.0, area)))
    raise ValueError(f"Unknown method: {method}")


def list_image_files(directory: str | os.PathLike) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    directory = Path(directory)
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in exts])
