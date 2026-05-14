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
  2>&1 | tee runs_logs/prepare_yolo11_defect_baseline.log


echo "=================================================="
echo "[2] Train YOLOv11 defect baseline - FULL"
echo "=================================================="

python scripts/train_yolo11_defect_baseline.py \
  --data data_yolo11_defect_baseline/data.yaml \
  --model yolo11m-seg.pt \
  --epochs 10 \
  --imgsz 640 \
  --batch 48 \
  --workers 16 \
  --fraction 1.0 \
  --name yolo11m_defect_640_full_10ep \
  2>&1 | tee runs_logs/train_yolo11m_defect_640_full_10ep.log


echo "=================================================="
echo "[3] Evaluate on Test set - FULL"
echo "=================================================="

python scripts/evaluate_yolo11_defect_baseline.py \
  --val-dir ./data/Test \
  --weights runs/baseline/yolo11m_defect_640_full_10ep/weights/best.pt \
  --output runs/baseline_eval_test_full_yolo11m_defect_640_10ep \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.5 \
  2>&1 | tee runs_logs/eval_test_full_yolo11m_defect_640_10ep.log


echo "=================================================="
echo "FULL EXPERIMENT DONE"
echo "=================================================="

date