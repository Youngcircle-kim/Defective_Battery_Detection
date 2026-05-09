"""Patch 결과 병합, 마스크 정제."""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


def update_max_conf_mask(
    global_conf: np.ndarray,
    global_label: np.ndarray,  # 0 = background, otherwise class_id+1
    patch_mask_bool: np.ndarray,
    patch_conf: float,
    class_id: int,
    offset_xy: Tuple[int, int],
) -> None:
    """
    patch 단위 binary mask + scalar confidence를 global mask에 max-conf 병합.
    여러 클래스 중 가장 높은 conf로 픽셀 ownership 결정.

    global_conf: (H, W) float
    global_label: (H, W) int  (0=BG, 1+ = class_id + 1 → label - 1로 환원)
    """
    ox, oy = offset_xy
    ph, pw = patch_mask_bool.shape
    H, W = global_conf.shape
    x1 = max(0, ox)
    y1 = max(0, oy)
    x2 = min(W, ox + pw)
    y2 = min(H, oy + ph)
    if x2 <= x1 or y2 <= y1:
        return
    sub_patch = patch_mask_bool[y1 - oy : y2 - oy, x1 - ox : x2 - ox]
    if not sub_patch.any():
        return
    sub_global_conf = global_conf[y1:y2, x1:x2]
    sub_global_lbl = global_label[y1:y2, x1:x2]
    update = sub_patch & (patch_conf > sub_global_conf)
    sub_global_conf[update] = patch_conf
    sub_global_lbl[update] = class_id + 1


def confine_to_battery(
    masks: Dict[str, np.ndarray], battery_mask: np.ndarray
) -> Dict[str, np.ndarray]:
    """damaged, pollution을 battery_outline 안에만 살림."""
    out = {}
    for k, m in masks.items():
        if k == "battery_outline":
            out[k] = m.astype(bool)
        else:
            out[k] = m.astype(bool) & battery_mask.astype(bool)
    return out


def remove_small_components(
    mask: np.ndarray, min_area: int
) -> np.ndarray:
    """connected component 중 작은 noise 제거."""
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lbl] = 1
    return out.astype(bool)


def morphology_clean(mask: np.ndarray, kernel: int) -> np.ndarray:
    """opening → closing 으로 noise 제거 + hole 메우기."""
    if kernel <= 1:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m.astype(bool)


def mask_to_polygons(mask: np.ndarray, epsilon_ratio: float = 0.002) -> list:
    """binary mask → list of polygon (px coords)."""
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polys = []
    for c in contours:
        if c.shape[0] < 3:
            continue
        eps = epsilon_ratio * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if approx.shape[0] >= 3:
            polys.append(approx.astype(int).tolist())
    return polys
