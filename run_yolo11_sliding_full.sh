#!/usr/bin/env bash
set -euo pipefail

# ==================================================
# YOLOv11 Sliding-Crop Baseline Full Experiment
# Classes:
#   0: battery_outline
#   1: damaged
#   2: pollution
# ==================================================

mkdir -p runs_logs

# -----------------------------
# Experiment settings
# -----------------------------
DATA_ROOT="data_yolo11_sliding_baseline"

PROJECT="runs/baseline"
EXP_NAME="yolo11n_sliding640_3class_10ep"

WEIGHTS="${PROJECT}/${EXP_NAME}/weights/best.pt"

FULL_EVAL_OUT="runs/sliding_eval_test_${EXP_NAME}_full"
VIS10_OUT="runs/sliding_eval_test_${EXP_NAME}_vis10"

CROP_SIZE=640
STRIDE=320
IMGSZ=640

MODEL="yolo11n-seg.pt"
EPOCHS=10
BATCH=128
WORKERS=16
DEVICE=0

CONF=0.25
IOU=0.5

echo "=================================================="
echo "[Settings]"
echo "DATA_ROOT      = ${DATA_ROOT}"
echo "PROJECT        = ${PROJECT}"
echo "EXP_NAME       = ${EXP_NAME}"
echo "WEIGHTS        = ${WEIGHTS}"
echo "MODEL          = ${MODEL}"
echo "EPOCHS         = ${EPOCHS}"
echo "BATCH          = ${BATCH}"
echo "CROP_SIZE      = ${CROP_SIZE}"
echo "STRIDE         = ${STRIDE}"
echo "IMGSZ          = ${IMGSZ}"
echo "DEVICE         = ${DEVICE}"
echo "=================================================="


echo "=================================================="
echo "[1] Prepare sliding-crop dataset"
echo "=================================================="

python scripts/prepare_yolo11_sliding_baseline.py \
  --config configs/config.yaml \
  --out "${DATA_ROOT}" \
  --crop-size "${CROP_SIZE}" \
  --stride "${STRIDE}" \
  --negative-ratio 1.0 \
  --min-polygon-area-px 9 \
  --jpeg-quality 90 \
  --workers 16 \
  --chunksize 8 \
  --include-test \
  --clear \
  2>&1 | tee "runs_logs/prepare_${EXP_NAME}.log"

echo "=================================================="
echo "[2] Train YOLOv11 sliding baseline"
echo "=================================================="

python scripts/train_yolo11_defect_baseline.py \
  --data "${DATA_ROOT}/data.yaml" \
  --model "${MODEL}" \
  --epochs "${EPOCHS}" \
  --imgsz "${IMGSZ}" \
  --batch "${BATCH}" \
  --workers "${WORKERS}" \
  --device "${DEVICE}" \
  --project "${PROJECT}" \
  --fraction 1.0 \
  --name "${EXP_NAME}" \
  2>&1 | tee "runs_logs/train_${EXP_NAME}.log"


echo "=================================================="
echo "[3] Check weight path"
echo "=================================================="

if [ ! -f "${WEIGHTS}" ]; then
  echo "[ERROR] Weight file not found: ${WEIGHTS}"
  echo "Available weights:"
  find "${PROJECT}" -path "*/weights/*.pt" | sort || true
  exit 1
fi

echo "[OK] Found weights: ${WEIGHTS}"


echo "=================================================="
echo "[4] Full Test Evaluation"
echo "=================================================="

python scripts/evaluate_yolo11_sliding_baseline.py \
  --val-dir ./data/Test \
  --weights "${WEIGHTS}" \
  --output "${FULL_EVAL_OUT}" \
  --crop-size "${CROP_SIZE}" \
  --stride "${STRIDE}" \
  --imgsz "${IMGSZ}" \
  --conf "${CONF}" \
  --iou "${IOU}" \
  --device "${DEVICE}" \
  2>&1 | tee "runs_logs/eval_full_${EXP_NAME}.log"


echo "=================================================="
echo "[5] Save first 10 visualization images"
echo "=================================================="

python scripts/evaluate_yolo11_sliding_baseline.py \
  --val-dir ./data/Test \
  --weights "${WEIGHTS}" \
  --output "${VIS10_OUT}" \
  --crop-size "${CROP_SIZE}" \
  --stride "${STRIDE}" \
  --imgsz "${IMGSZ}" \
  --conf "${CONF}" \
  --iou "${IOU}" \
  --device "${DEVICE}" \
  --limit 10 \
  --save-vis \
  2>&1 | tee "runs_logs/eval_vis10_${EXP_NAME}.log"


echo "=================================================="
echo "[6] Summary"
echo "=================================================="

echo "Weights:"
echo "${WEIGHTS}"

echo ""
echo "Full eval summary:"
cat "${FULL_EVAL_OUT}/metrics_summary.json"

echo ""
echo "Visualization folder:"
echo "${VIS10_OUT}/visualizations"

echo ""
echo "DONE"
date