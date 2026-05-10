"""
Validation set 전체에 대해 segmentation metric 평가.

원본 JSON 라벨(GT)을 클래스별 binary mask로 렌더링한 뒤
파이프라인 예측 mask와 비교 → IoU, F1, Pixel Accuracy, mIoU.
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


def gt_masks_from_label(
    label_path: Path, w: int, h: int, schema: dict
) -> Dict[str, np.ndarray]:
    """JSON 라벨 → 클래스명별 binary mask (절대 픽셀 좌표 사용)."""
    masks = {
        "battery_outline": np.zeros((h, w), dtype=bool),
        "damaged":         np.zeros((h, w), dtype=bool),  # large+small 통합
        "pollution":       np.zeros((h, w), dtype=bool),
    }
    if not label_path.exists():
        return masks

    try:
        parsed = parse_json_label(label_path, schema)
    except Exception as e:
        print(f"[warn] JSON 파싱 실패 {label_path.name}: {e}")
        return masks

    for cls_name, polys in parsed.items():
        for poly_px in polys:
            poly_int = poly_px.astype(np.int32)
            layer = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(layer, [poly_int], 1)
            masks[cls_name] |= layer.astype(bool)
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--val-dir", type=str, default=None,
                    help="원본 Validation 디렉토리 (None이면 config의 raw_val_*)")
    ap.add_argument("--output", type=str, default="runs/eval")
    ap.add_argument("--limit", type=int, default=0,
                    help=">0이면 처음 N장만 평가 (디버그용)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.val_dir:
        val_root = Path(args.val_dir)
        img_dir = val_root / "image_data" / "images"
        lbl_dir = val_root / "label_data" / "labels"
    else:
        img_dir = Path(cfg["paths"]["raw_val_images"])
        lbl_dir = Path(cfg["paths"]["raw_val_labels"])

    pipe = TwoStageBatteryPipeline(
        stage1_weights=cfg["inference"]["stage1_weights"],
        stage2_weights=cfg["inference"]["stage2_weights"],
        device=cfg["stage1"]["device"],
        roi_margin=cfg["inference"]["roi_margin"],
        patch_size=cfg["stage2"]["patch_size"],
        patch_stride=cfg["stage2"]["patch_stride"],
        stage_imgsz=cfg["stage1"]["imgsz"],
        conf_threshold=cfg["inference"]["conf_threshold"],
        iou_threshold=cfg["inference"]["iou_threshold"],
        morph_kernel=cfg["inference"]["morph_kernel"],
        morph_min_area_px=cfg["inference"]["morph_min_area_px"],
    )

    schema = cfg["json_schema"]
    img_paths = list_image_files(img_dir)
    if args.limit > 0:
        img_paths = img_paths[: args.limit]
    print(f"평가 대상: {len(img_paths)}장")

    sample_metrics = []
    per_image_log = []

    for img_path in tqdm(img_paths, desc="evaluate"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = gt_masks_from_label(
            lbl_dir / (img_path.stem + ".json"), w, h, schema,
        )
        pred = pipe.predict(img)

        m = evaluate_sample(pred, gt)
        sample_metrics.append(m)
        per_image_log.append({"image": img_path.name, "metrics": m})

    agg = aggregate_metrics(sample_metrics)

    # 콘솔 출력
    print("\n========== Aggregated Metrics ==========")
    for cls, sub in agg.items():
        print(f"[{cls}]")
        for k, v in sub.items():
            print(f"   {k:>16}: {v:.4f}")

    # JSON으로 저장
    with open(out / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    with open(out / "metrics_per_image.json", "w", encoding="utf-8") as f:
        json.dump(per_image_log, f, indent=2, ensure_ascii=False)
    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()
