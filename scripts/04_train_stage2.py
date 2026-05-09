"""Stage 2 (Patch) 학습."""
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
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    s2 = cfg["stage2"]

    data_yaml = Path(cfg["paths"]["stage2_patches_root"]) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} 없음. 먼저 scripts/02_generate_patches.py 실행."
        )

    model = YOLO(s2["model"])
    model.train(
        data=str(data_yaml),
        epochs=s2["epochs"],
        imgsz=s2["imgsz"],
        batch=s2["batch"],
        device=s2["device"],
        patience=s2["patience"],
        optimizer=s2["optimizer"],
        lr0=s2["lr0"],
        project=s2["project"],
        name=s2["name"],
        exist_ok=True,
        resume=args.resume,
        amp=True,
        cache=False,
        workers=4,
    )
    print("Stage2 학습 완료.")


if __name__ == "__main__":
    main()
