"""Stage 1 quick training - fast pilot version."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--model", type=str, default="yolo11s-seg.pt")
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--name", type=str, default="stage1_fast_pilot")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    s1 = cfg["stage1"]

    if args.data is None:
        data_yaml = Path(cfg["paths"]["stage1_root"]) / "data.yaml"
    else:
        data_yaml = Path(args.data)

    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} 없음.")

    model = YOLO(args.model)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=s1["device"],
        patience=max(3, min(5, args.epochs)),
        optimizer=s1["optimizer"],
        lr0=s1["lr0"],
        project=s1["project"],
        name=args.name,
        exist_ok=True,
        resume=args.resume,

        # speed
        amp=True,
        cache=False,
        workers=args.workers,
        fraction=args.fraction,
        freeze=10,

        # lighter augmentation for pilot
        cos_lr=True,
        close_mosaic=min(3, args.epochs),
        mosaic=0.3,
        mixup=0.0,
        copy_paste=0.0,

        plots=True,
        save=True,
        save_period=-1,
        verbose=True,
    )

    print("Stage1 fast pilot training complete.")


if __name__ == "__main__":
    main()