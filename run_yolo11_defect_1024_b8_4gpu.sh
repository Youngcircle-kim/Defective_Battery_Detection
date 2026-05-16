#!/usr/bin/env bash
set -e

mkdir -p runs_logs

echo "=================================================="
echo "[1] Prepare YOLOv11 defect baseline dataset"
echo "=================================================="

python scripts/prepare_yolo11_defect_baseline.py \
  --config configs/config.yaml \
  --out data_yolo11_defect_baseline \
  --mode symlink \
  --include-test \
  --clear \
  2>&1 | tee runs_logs/prepare_yolo11_defect_baseline_yolo11n_1024.log


echo "=================================================="
echo "[2] Train YOLOv11n-Seg defect baseline - imgsz 1024 / batch 8 / 4 GPUs"
echo "=================================================="

python scripts/train_yolo11_defect_baseline.py \
  --data data_yolo11_defect_baseline/data.yaml \
  --model yolo11n-seg.pt \
  --epochs 100 \
  --imgsz 1024 \
  --batch 8 \
  --workers 16 \
  --device 0,1,2,3 \
  --fraction 1.0 \
  --name yolo11n_defect_1024_b8_4gpu_100ep \
  2>&1 | tee runs_logs/train_yolo11n_defect_1024_b8_4gpu_100ep.log


echo "=================================================="
echo "[3] Evaluate on Test set - YOLOv11n 1024"
echo "=================================================="

python scripts/evaluate_yolo11_defect_baseline.py \
  --val-dir ./data/Test \
  --weights runs/baseline/yolo11n_defect_1024_b8_4gpu_100ep/weights/best.pt \
  --output runs/baseline_eval_test_yolo11n_defect_1024_b8_4gpu_100ep \
  --imgsz 1024 \
  --conf 0.25 \
  --iou 0.5 \
  2>&1 | tee runs_logs/eval_test_yolo11n_defect_1024_b8_4gpu_100ep.log


echo "=================================================="
echo "YOLOv11n 1024 / batch 8 / 4GPU EXPERIMENT DONE"
echo "=================================================="

date
