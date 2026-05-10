"""
JSON 라벨 → 표준 polygon 형식 변환기.

실제 데이터셋 스키마 (확인됨):
{
  "data_info": {...},
  "swelling": {
    "swelling": false,
    "battery_outline": [x1, y1, x2, y2, ...]    # flat list, 절대 픽셀
  },
  "defects": [
    {"id": ..., "name": "Damaged" | "Pollution",
     "points": [x1, y1, x2, y2, ...]}            # flat list, 절대 픽셀
  ],
  "image_info": {
    "width": 1920, "height": 1080,
    "file_name": "...png", "id": ..., "is_normal": bool
  }
}

* polygon 좌표가 nested ([[x,y],...]), flat ([x,y,x,y,...]),
  홀수 길이(끝에 점 1개 누락) 모두 자동 처리.
* 키 이름은 configs/config.yaml의 json_schema 섹션에서 변경 가능.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _coerce_polygon(raw) -> np.ndarray | None:
    """
    polygon 좌표를 (N, 2) np.ndarray로 변환.
    지원 형식:
      - [[x,y],[x,y],...]  (nested)
      - [x1,y1,x2,y2,...]  (flat, 짝수)
      - [x1,y1,...,xn]     (flat, 홀수 → 마지막 값 버리고 처리)
      - {"x": [...], "y": [...]} (xy split)
    """
    if raw is None:
        return None

    # dict 형태
    if isinstance(raw, dict):
        if "x" in raw and "y" in raw:
            xs = np.asarray(raw["x"], dtype=np.float32)
            ys = np.asarray(raw["y"], dtype=np.float32)
            if xs.shape != ys.shape or xs.size < 3:
                return None
            return np.stack([xs, ys], axis=1)
        for key in ("polygon", "points", "coords", "segmentation"):
            if key in raw:
                return _coerce_polygon(raw[key])
        return None

    if not isinstance(raw, (list, tuple)) or len(raw) == 0:
        return None

    # nested
    if isinstance(raw[0], (list, tuple)) and len(raw[0]) == 2:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
            return None
        return arr

    # flat
    if isinstance(raw[0], (int, float)):
        if len(raw) < 6:
            return None
        # 홀수 길이 방어: 마지막 값 버리고 진행 (데이터 노이즈 대응)
        if len(raw) % 2 != 0:
            raw = list(raw)[:-1]
        arr = np.asarray(raw, dtype=np.float32).reshape(-1, 2)
        if arr.shape[0] < 3:
            return None
        return arr

    return None


def _get_nested(d: dict, dotted: str):
    """'image_info.battery_outline.polygon' 같은 경로로 dict 접근."""
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def parse_json_label(
    json_path: str | Path,
    schema: Dict[str, str],
) -> Dict[str, List[np.ndarray]]:
    """
    JSON 라벨 파일을 읽고 표준 polygon dict 반환.

    Args:
        json_path: 라벨 JSON 파일 경로
        schema: configs/config.yaml의 json_schema 섹션

    Returns:
        {
          "battery_outline": [poly_px, ...],
          "damaged":         [poly_px, ...],
          "pollution":       [poly_px, ...],
        }
        각 poly_px는 (N, 2) float32, 절대 픽셀 좌표.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[str, List[np.ndarray]] = {
        "battery_outline": [],
        "damaged": [],
        "pollution": [],
    }

    # --- battery_outline
    bo_raw = _get_nested(data, schema["battery_outline_path"])
    bo_poly = _coerce_polygon(bo_raw)
    if bo_poly is not None:
        out["battery_outline"].append(bo_poly)

    # --- defects[]
    defects = _get_nested(data, schema["defects_array_path"]) or []
    name_key = schema["defect_name_key"]
    poly_key = schema["defect_polygon_key"]
    damaged_name = schema["damaged_name"].lower()
    pollution_name = schema["pollution_name"].lower()

    for d in defects:
        if not isinstance(d, dict):
            continue
        name = str(d.get(name_key, "")).lower()
        poly = _coerce_polygon(d.get(poly_key))
        if poly is None:
            continue
        if name == damaged_name:
            out["damaged"].append(poly)
        elif name == pollution_name:
            out["pollution"].append(poly)

    return out


def get_image_size(json_path: str | Path, schema: Dict[str, str]) -> Tuple[int, int] | None:
    """JSON에서 (width, height) 가져오기. 없으면 None."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    w = _get_nested(data, schema.get("image_width_path", "image_info.width"))
    h = _get_nested(data, schema.get("image_height_path", "image_info.height"))
    if w is None or h is None:
        return None
    try:
        return int(w), int(h)
    except (TypeError, ValueError):
        return None


def is_normal_image(json_path: str | Path, schema: Dict[str, str]) -> bool | None:
    """JSON의 is_normal 플래그. 없으면 None."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    val = _get_nested(data, schema.get("is_normal_path", "image_info.is_normal"))
    if val is None:
        return None
    return bool(val)
