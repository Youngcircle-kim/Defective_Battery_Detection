"""Stage 2 (Patch) quick/fast training script.

- Uses Stage2 patch dataset
- Supports fraction for quick pilot experiments
- Allows overriding epochs, batch, workers, model, imgsz from CLI
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
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--data", type=str, default=None)

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--model", type=str, default=None)

    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--name", type=str, default="stage2_fast_pilot")
    ap.add_argument("--resume", action="store_true")

    # Optional speed/stability options
    ap.add_argument("--freeze", type=int, default=10)
    ap.add_argument("--cache", action="store_true")

    args = ap.parse_args()

    cfg = load_config(args.config)
    s2 = cfg["stage2"]

    if args.data is None:
        data_yaml = Path(cfg["paths"]["stage2_patches_root"]) / "data.yaml"
    else:
        data_yaml = Path(args.data)

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} 없음. 먼저 scripts/02_generate_patches.py 실행."
        )

    model_path = args.model if args.model is not None else s2["model"]
    imgsz = args.imgsz if args.imgsz is not None else s2["imgsz"]

    model = YOLO(model_path)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=imgsz,
        batch=args.batch,
        device=s2["device"],
        patience=max(3, min(5, args.epochs)),
        optimizer=s2["optimizer"],
        lr0=s2["lr0"],
        project=s2["project"],
        name=args.name,
        exist_ok=True,
        resume=args.resume,

        # Speed
        amp=True,
        cache=args.cache,
        workers=args.workers,
        fraction=args.fraction,
        freeze=args.freeze,

        # Light augmentation for pilot
        cos_lr=True,
        close_mosaic=min(3, args.epochs),
        mosaic=0.3,
        mixup=0.0,
        copy_paste=0.0,

        # Logging
        plots=True,
        save=True,
        save_period=-1,
        verbose=True,
    )

    print("Stage2 fast pilot training complete.")


if __name__ == "__main__":
    main()