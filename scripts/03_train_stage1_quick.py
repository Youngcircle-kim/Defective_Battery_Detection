"""Stage 1 quick experiment training script.

- Uses sampled data.yaml
- Allows short epochs for pilot experiments
- Optimized for A100 MIG 3g.40gb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data", type=str, default="data_stage1_sample/data.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--name", type=str, default="stage1_quick")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    s1 = cfg["stage1"]

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} 없음. 먼저 sample dataset을 생성하세요."
        )

    model = YOLO(s1["model"])

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=s1["imgsz"],
        batch=args.batch,
        device=s1["device"],
        patience=max(3, min(5, args.epochs)),
        optimizer=s1["optimizer"],
        lr0=s1["lr0"],
        project=s1["project"],
        name=args.name,
        exist_ok=True,
        resume=args.resume,

        amp=True,
        cache=False,
        workers=args.workers,

        cos_lr=True,
        close_mosaic=min(3, args.epochs),

        plots=True,
        save=True,
        save_period=-1,
        verbose=True,
    )

    print("Stage1 quick experiment complete.")


if __name__ == "__main__":
    main()