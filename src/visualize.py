"""결과 시각화."""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


def overlay_masks(
    image: np.ndarray,
    masks: Dict[str, np.ndarray],
    colors: Dict[str, Tuple[int, int, int]],
    alpha: float = 0.45,
) -> np.ndarray:
    """이미지에 클래스별 마스크 컬러 오버레이. image: BGR uint8."""
    out = image.copy()
    for cls, mask in masks.items():
        color = colors.get(cls)
        if color is None:
            continue
        m = mask.astype(bool)
        if not m.any():
            continue
        overlay = np.zeros_like(out)
        overlay[m] = color
        out = cv2.addWeighted(out, 1.0, overlay, alpha, 0)

    # 클래스 외곽선 추가 (가독성)
    for cls, mask in masks.items():
        color = colors.get(cls)
        if color is None or not mask.any():
            continue
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, thickness=2)
    return out


def draw_legend(
    image: np.ndarray, colors: Dict[str, Tuple[int, int, int]]
) -> np.ndarray:
    """좌상단에 클래스 legend 그리기."""
    out = image.copy()
    x, y = 12, 12
    box_h = 22
    for cls, color in colors.items():
        cv2.rectangle(out, (x, y), (x + 18, y + 18), color, -1)
        cv2.rectangle(out, (x, y), (x + 18, y + 18), (255, 255, 255), 1)
        cv2.putText(
            out, cls, (x + 26, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            out, cls, (x + 26, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )
        y += box_h + 4
    return out
