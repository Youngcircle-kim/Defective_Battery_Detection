"""Train single-stage YOLOv11-Seg defect baseline.

Classes:
  0: damaged
  1: pollution
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="data_yolo11_defect_baseline/data.yaml")
    parser.add_argument("--model", type=str, default="yolo11m-seg.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/baseline")
    parser.add_argument("--name", type=str, default="yolo11_defect_baseline")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--cache", action="store_true")

    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    model = YOLO(args.model)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,

        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=max(3, min(10, args.epochs)),
        amp=True,
        cache=args.cache,
        cos_lr=True,
        close_mosaic=min(10, args.epochs),

        fraction=args.fraction,

        plots=True,
        save=True,
        save_period=-1,
        verbose=True,
    )

    if args.freeze > 0:
        train_kwargs["freeze"] = args.freeze

    model.train(**train_kwargs)

    print("YOLOv11 defect baseline training complete.")


if __name__ == "__main__":
    main()