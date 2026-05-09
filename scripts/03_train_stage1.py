"""Stage 1 (Global) 학습."""
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
    s1 = cfg["stage1"]

    data_yaml = Path(cfg["paths"]["stage1_root"]) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} 없음. 먼저 scripts/01_split_labels.py 실행."
        )

    model = YOLO(s1["model"])
    model.train(
        data=str(data_yaml),
        epochs=s1["epochs"],
        imgsz=s1["imgsz"],
        batch=s1["batch"],
        device=s1["device"],
        patience=s1["patience"],
        optimizer=s1["optimizer"],
        lr0=s1["lr0"],
        project=s1["project"],
        name=s1["name"],
        exist_ok=True,
        resume=args.resume,
        # GTX 3060 메모리 절약
        amp=True,
        cache=False,
        workers=4,
    )
    print("Stage1 학습 완료.")


if __name__ == "__main__":
    main()
