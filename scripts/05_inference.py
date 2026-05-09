"""전체 파이프라인 추론 (단일 이미지 또는 디렉토리)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import list_image_files, load_config
from src.pipeline import TwoStageBatteryPipeline
from src.postprocess import mask_to_polygons
from src.visualize import draw_legend, overlay_masks


def run_inference(cfg: dict, input_path: Path, output_dir: Path):
    inf = cfg["inference"]
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = TwoStageBatteryPipeline(
        stage1_weights=inf["stage1_weights"],
        stage2_weights=inf["stage2_weights"],
        device=cfg["stage1"]["device"],
        roi_margin=inf["roi_margin"],
        patch_size=cfg["stage2"]["patch_size"],
        patch_stride=cfg["stage2"]["patch_stride"],
        stage_imgsz=cfg["stage1"]["imgsz"],
        conf_threshold=inf["conf_threshold"],
        iou_threshold=inf["iou_threshold"],
        morph_kernel=inf["morph_kernel"],
        morph_min_area_px=inf["morph_min_area_px"],
    )

    if input_path.is_dir():
        images = list_image_files(input_path)
    else:
        images = [input_path]

    colors = {k: tuple(v) for k, v in cfg["viz"]["colors"].items()}
    alpha = cfg["viz"]["alpha"]

    for img_path in tqdm(images, desc="inference"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"skip (read fail): {img_path}")
            continue

        masks = pipe.predict(img)

        # 시각화
        vis = overlay_masks(img, masks, colors, alpha=alpha)
        vis = draw_legend(vis, colors)
        cv2.imwrite(str(output_dir / f"{img_path.stem}_visualization.png"), vis)

        # binary mask 저장
        mask_dir = output_dir / f"{img_path.stem}_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        for cls, m in masks.items():
            cv2.imwrite(str(mask_dir / f"{cls}.png"), (m.astype("uint8") * 255))

        # polygon 좌표 저장
        polygons = {cls: mask_to_polygons(m) for cls, m in masks.items()}
        with open(output_dir / f"{img_path.stem}_polygons.json", "w", encoding="utf-8") as f:
            json.dump({
                "image": img_path.name,
                "image_size": {"width": img.shape[1], "height": img.shape[0]},
                "polygons": polygons,
            }, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--image", type=str, required=True,
                    help="이미지 파일 또는 디렉토리")
    ap.add_argument("--output", type=str, default="runs/inference")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_inference(cfg, Path(args.image), Path(args.output))
    print(f"결과 저장 완료: {args.output}")


if __name__ == "__main__":
    main()
