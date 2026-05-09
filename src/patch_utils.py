"""ROI 추출, patch 생성, polygon 클리핑."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid


def compute_roi_bbox(
    battery_polygons_px: List[np.ndarray],
    img_w: int,
    img_h: int,
    margin: float = 0.1,
) -> Tuple[int, int, int, int]:
    """
    battery_outline polygon들 합집합의 bbox + margin.

    Returns (x1, y1, x2, y2) in image pixel coords, clipped to image.
    Battery가 없으면 전체 이미지 반환.
    """
    if not battery_polygons_px:
        return 0, 0, img_w, img_h

    all_pts = np.concatenate(battery_polygons_px, axis=0)
    x1 = float(all_pts[:, 0].min())
    y1 = float(all_pts[:, 1].min())
    x2 = float(all_pts[:, 0].max())
    y2 = float(all_pts[:, 1].max())

    bw = x2 - x1
    bh = y2 - y1
    mx = bw * margin
    my = bh * margin

    x1 = int(max(0, np.floor(x1 - mx)))
    y1 = int(max(0, np.floor(y1 - my)))
    x2 = int(min(img_w, np.ceil(x2 + mx)))
    y2 = int(min(img_h, np.ceil(y2 + my)))
    return x1, y1, x2, y2


def generate_patch_grid(
    roi_w: int,
    roi_h: int,
    patch_size: int,
    stride: int,
) -> List[Tuple[int, int]]:
    """
    ROI 안에서 patch 좌상단 좌표 리스트 생성.
    가장자리 커버 보장 (마지막 patch가 ROI 끝에 닿도록).
    """
    if roi_w < patch_size or roi_h < patch_size:
        # ROI가 patch보다 작으면 padding 위치 (0,0) 하나만
        return [(0, 0)]

    xs = list(range(0, roi_w - patch_size + 1, stride))
    if xs[-1] != roi_w - patch_size:
        xs.append(roi_w - patch_size)
    ys = list(range(0, roi_h - patch_size + 1, stride))
    if ys[-1] != roi_h - patch_size:
        ys.append(roi_h - patch_size)
    return [(x, y) for y in ys for x in xs]


def crop_patch(
    image: np.ndarray, x: int, y: int, patch_size: int
) -> np.ndarray:
    """ROI 이미지에서 patch crop. patch_size보다 작으면 검은색 padding."""
    h, w = image.shape[:2]
    xe = min(x + patch_size, w)
    ye = min(y + patch_size, h)
    crop = image[y:ye, x:xe]
    pad_h = patch_size - crop.shape[0]
    pad_w = patch_size - crop.shape[1]
    if pad_h > 0 or pad_w > 0:
        crop = np.pad(
            crop,
            ((0, pad_h), (0, pad_w), (0, 0)) if image.ndim == 3 else ((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=0,
        )
    return crop


def _to_polygons(geom: BaseGeometry) -> List[ShapelyPolygon]:
    """geom을 list of Polygon으로 풀어서 반환."""
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        out = []
        for g in geom.geoms:
            out.extend(_to_polygons(g))
        return out
    return []


def clip_polygon_to_box(
    polygon_px: np.ndarray,
    box_xyxy: Tuple[float, float, float, float],
    min_area: float = 1.0,
) -> List[np.ndarray]:
    """
    polygon을 box 영역에 클리핑. 결과는 박스의 (0,0) 기준 local 좌표.
    여러 조각으로 나뉘면 각각 반환.
    """
    if polygon_px.shape[0] < 3:
        return []
    x1, y1, x2, y2 = box_xyxy
    poly = ShapelyPolygon(polygon_px)
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty:
        return []
    clip = poly.intersection(shapely_box(x1, y1, x2, y2))
    pieces = _to_polygons(clip)

    out: List[np.ndarray] = []
    for p in pieces:
        if p.area < min_area:
            continue
        coords = np.array(p.exterior.coords, dtype=np.float32)[:-1]  # close 중복 제거
        if coords.shape[0] < 3:
            continue
        local = coords.copy()
        local[:, 0] -= x1
        local[:, 1] -= y1
        out.append(local)
    return out
