"""
Validation/Test set 전체에 대해 segmentation metric 평가.

원본 JSON 라벨(GT)을 클래스별 binary mask로 렌더링한 뒤
파이프라인 예측 mask와 비교 → IoU, F1, Pixel Accuracy, mIoU.

Pilot / full experiment 둘 다 지원:
- --stage1-weights
- --stage2-weights
- --imgsz
- --conf
- --iou
- --val-dir 또는 --image-dir / --label-dir 직접 지정
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import list_image_files, load_config
from src.json_label_loader import parse_json_label
from src.metrics import aggregate_metrics, evaluate_sample
from src.pipeline import TwoStageBatteryPipeline


def find_label_path(label_dir: Path, img_path: Path) -> Path | None:
    """
    이미지 파일명에 대응하는 JSON 라벨 찾기.

    기본:
      labels/RGB_xxx.json

    fallback:
      하위 폴더 구조가 있는 경우를 대비해 stem 기준 검색
    """
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
    """JSON 라벨 → 클래스명별 binary mask."""
    masks = {
        "battery_outline": np.zeros((h, w), dtype=bool),
        "damaged": np.zeros((h, w), dtype=bool),
        "pollution": np.zeros((h, w), dtype=bool),
    }

    if label_path is None or not label_path.exists():
        return masks

    try:
        parsed = parse_json_label(label_path, schema)
    except Exception as e:
        print(f"[warn] JSON 파싱 실패 {label_path}: {e}")
        return masks

    # parse_json_label의 반환 구조:
    # {
    #   "battery_outline": [...],
    #   "damaged": [...],
    #   "pollution": [...]
    # }
    for cls_name, polys in parsed.items():
        if cls_name not in masks:
            continue

        for poly_px in polys:
            if poly_px is None or len(poly_px) < 3:
                continue

            poly_int = poly_px.astype(np.int32)
            layer = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(layer, [poly_int], 1)
            masks[cls_name] |= layer.astype(bool)

    return masks


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", type=str, default="configs/config.yaml")

    # 1) 기존 방식: data/Validation 같은 root 입력
    ap.add_argument(
        "--val-dir",
        type=str,
        default=None,
        help="Validation/Test root. 예: data/Validation 또는 data/Test",
    )

    # 2) 직접 이미지/라벨 폴더 지정
    ap.add_argument("--image-dir", type=str, default=None)
    ap.add_argument("--label-dir", type=str, default=None)

    ap.add_argument("--output", type=str, default="runs/eval")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help=">0이면 처음 N장만 평가",
    )

    # pilot weight override
    ap.add_argument("--stage1-weights", type=str, default=None)
    ap.add_argument("--stage2-weights", type=str, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=None)

    args = ap.parse_args()

    cfg = load_config(args.config)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # 평가 데이터 경로 결정
    if args.image_dir is not None:
        img_dir = Path(args.image_dir)

        if args.label_dir is None:
            raise ValueError("--image-dir를 쓰면 --label-dir도 같이 지정해야 합니다.")
        lbl_dir = Path(args.label_dir)

    elif args.val_dir is not None:
        val_root = Path(args.val_dir)
        img_dir = val_root / "image_data" / "images"
        lbl_dir = val_root / "label_data" / "labels"

    else:
        img_dir = Path(cfg["paths"]["raw_val_images"])
        lbl_dir = Path(cfg["paths"]["raw_val_labels"])

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not lbl_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {lbl_dir}")

    inf = cfg["inference"]

    stage1_weights = args.stage1_weights or inf["stage1_weights"]
    stage2_weights = args.stage2_weights or inf["stage2_weights"]
    stage_imgsz = args.imgsz if args.imgsz is not None else cfg["stage1"]["imgsz"]
    conf_threshold = args.conf if args.conf is not None else inf["conf_threshold"]
    iou_threshold = args.iou if args.iou is not None else inf["iou_threshold"]

    stage1_weights = Path(stage1_weights)
    stage2_weights = Path(stage2_weights)

    if not stage1_weights.exists():
        raise FileNotFoundError(f"Stage1 weight not found: {stage1_weights}")
    if not stage2_weights.exists():
        raise FileNotFoundError(f"Stage2 weight not found: {stage2_weights}")

    print("[Evaluation Config]")
    print(f"  image_dir      : {img_dir}")
    print(f"  label_dir      : {lbl_dir}")
    print(f"  output         : {out}")
    print(f"  stage1_weights : {stage1_weights}")
    print(f"  stage2_weights : {stage2_weights}")
    print(f"  imgsz          : {stage_imgsz}")
    print(f"  conf           : {conf_threshold}")
    print(f"  iou            : {iou_threshold}")
    print(f"  patch_size     : {cfg['stage2']['patch_size']}")
    print(f"  patch_stride   : {cfg['stage2']['patch_stride']}")

    pipe = TwoStageBatteryPipeline(
        stage1_weights=str(stage1_weights),
        stage2_weights=str(stage2_weights),
        device=cfg["stage1"]["device"],
        roi_margin=inf["roi_margin"],
        patch_size=cfg["stage2"]["patch_size"],
        patch_stride=cfg["stage2"]["patch_stride"],
        stage_imgsz=stage_imgsz,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        morph_kernel=inf["morph_kernel"],
        morph_min_area_px=inf["morph_min_area_px"],
    )

    schema = cfg["json_schema"]

    img_paths = list_image_files(img_dir)
    if args.limit > 0:
        img_paths = img_paths[: args.limit]

    print(f"평가 대상: {len(img_paths)}장")

    sample_metrics = []
    per_image_log = []
    missing_labels = 0

    for img_path in tqdm(img_paths, desc="evaluate"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warn] 이미지 읽기 실패: {img_path}")
            continue

        h, w = img.shape[:2]

        label_path = find_label_path(lbl_dir, img_path)
        if label_path is None:
            missing_labels += 1

        gt = gt_masks_from_label(label_path, w, h, schema)
        pred = pipe.predict(img)

        m = evaluate_sample(pred, gt)
        sample_metrics.append(m)

        per_image_log.append(
            {
                "image": img_path.name,
                "label": None if label_path is None else str(label_path),
                "metrics": m,
            }
        )

    if not sample_metrics:
        raise RuntimeError("평가 가능한 이미지가 없습니다.")

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
                    "stage1_weights": str(stage1_weights),
                    "stage2_weights": str(stage2_weights),
                    "imgsz": stage_imgsz,
                    "conf": conf_threshold,
                    "iou": iou_threshold,
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

    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()