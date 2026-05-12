"""전체 파이프라인 추론 (단일 이미지 또는 디렉토리).

Pilot / full experiment 둘 다 지원:
- --stage1-weights
- --stage2-weights
- --imgsz
- --conf
- --iou
"""

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


def run_inference(
    cfg: dict,
    input_path: Path,
    output_dir: Path,
    stage1_weights: str | None = None,
    stage2_weights: str | None = None,
    imgsz: int | None = None,
    conf: float | None = None,
    iou: float | None = None,
):
    inf = cfg["inference"]
    output_dir.mkdir(parents=True, exist_ok=True)

    s1_weights = stage1_weights or inf["stage1_weights"]
    s2_weights = stage2_weights or inf["stage2_weights"]

    stage_imgsz = imgsz if imgsz is not None else cfg["stage1"]["imgsz"]
    conf_threshold = conf if conf is not None else inf["conf_threshold"]
    iou_threshold = iou if iou is not None else inf["iou_threshold"]

    s1_weights = Path(s1_weights)
    s2_weights = Path(s2_weights)

    if not s1_weights.exists():
        raise FileNotFoundError(f"Stage1 weight not found: {s1_weights}")
    if not s2_weights.exists():
        raise FileNotFoundError(f"Stage2 weight not found: {s2_weights}")

    print("[Inference Config]")
    print(f"  Stage1 weights : {s1_weights}")
    print(f"  Stage2 weights : {s2_weights}")
    print(f"  imgsz          : {stage_imgsz}")
    print(f"  conf           : {conf_threshold}")
    print(f"  iou            : {iou_threshold}")
    print(f"  patch_size     : {cfg['stage2']['patch_size']}")
    print(f"  patch_stride   : {cfg['stage2']['patch_stride']}")

    pipe = TwoStageBatteryPipeline(
        stage1_weights=str(s1_weights),
        stage2_weights=str(s2_weights),
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

    if input_path.is_dir():
        images = list_image_files(input_path)
    else:
        images = [input_path]

    if not images:
        raise RuntimeError(f"No images found: {input_path}")

    colors = {k: tuple(v) for k, v in cfg["viz"]["colors"].items()}
    alpha = cfg["viz"]["alpha"]

    for img_path in tqdm(images, desc="inference"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"skip (read fail): {img_path}")
            continue

        masks = pipe.predict(img)

        # 시각화 저장
        vis = overlay_masks(img, masks, colors, alpha=alpha)
        vis = draw_legend(vis, colors)
        cv2.imwrite(str(output_dir / f"{img_path.stem}_visualization.png"), vis)

        # binary mask 저장
        mask_dir = output_dir / f"{img_path.stem}_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

        for cls, m in masks.items():
            cv2.imwrite(
                str(mask_dir / f"{cls}.png"),
                (m.astype("uint8") * 255),
            )

        # polygon 좌표 저장
        polygons = {cls: mask_to_polygons(m) for cls, m in masks.items()}
        with open(
            output_dir / f"{img_path.stem}_polygons.json",
            "w",
            encoding="utf-8",
        ) as f:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument(
        "--image",
        type=str,
        required=True,
        help="이미지 파일 또는 디렉토리",
    )
    ap.add_argument("--output", type=str, default="runs/inference")

    ap.add_argument("--stage1-weights", type=str, default=None)
    ap.add_argument("--stage2-weights", type=str, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=None)

    args = ap.parse_args()

    cfg = load_config(args.config)

    run_inference(
        cfg=cfg,
        input_path=Path(args.image),
        output_dir=Path(args.output),
        stage1_weights=args.stage1_weights,
        stage2_weights=args.stage2_weights,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )

    print(f"결과 저장 완료: {args.output}")


if __name__ == "__main__":
    main()