"""Segmentation 평가 지표 (IoU, Pixel Accuracy, F1)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def per_class_iou(
    pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-9
) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")  # 둘 다 비어있으면 정의되지 않음
    return float(inter / (union + eps))


def per_class_f1(
    pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-9
) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    if tp + fp + fn == 0:
        return float("nan")
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    if precision + recall < eps:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def pixel_accuracy(
    pred_label: np.ndarray, gt_label: np.ndarray
) -> float:
    """multi-class label map (int)에서 픽셀 정확도."""
    return float((pred_label == gt_label).mean())


def evaluate_sample(
    pred_masks: Dict[str, np.ndarray],
    gt_masks: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    클래스별 IoU / F1, 그리고 pixel accuracy 계산.
    pred_masks/gt_masks는 같은 키(클래스명) 사용. shape 동일해야 함.
    """
    out: Dict[str, Dict[str, float]] = {}
    for cls, pred in pred_masks.items():
        gt = gt_masks.get(cls, np.zeros_like(pred, dtype=bool))
        out[cls] = {
            "iou": per_class_iou(pred, gt),
            "f1": per_class_f1(pred, gt),
        }

    # multi-class label (mutually exclusive 가정 안 함 — 그래도 픽셀 일치도)
    H, W = next(iter(pred_masks.values())).shape
    pred_lbl = np.zeros((H, W), dtype=np.int32)
    gt_lbl = np.zeros((H, W), dtype=np.int32)
    for i, cls in enumerate(pred_masks.keys(), start=1):
        pred_lbl[pred_masks[cls].astype(bool)] = i
        gt_lbl[gt_masks.get(cls, np.zeros((H, W), dtype=bool)).astype(bool)] = i
    out["__overall__"] = {"pixel_accuracy": pixel_accuracy(pred_lbl, gt_lbl)}
    return out


def aggregate_metrics(
    sample_metrics: List[Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """샘플별 metric → 평균. NaN 무시."""
    if not sample_metrics:
        return {}
    keys = sample_metrics[0].keys()
    agg: Dict[str, Dict[str, float]] = {}
    for k in keys:
        sub_keys = sample_metrics[0][k].keys()
        agg[k] = {}
        for sk in sub_keys:
            vals = [s[k][sk] for s in sample_metrics if sk in s.get(k, {})]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            agg[k][sk] = float(np.mean(vals)) if vals else float("nan")
    # mIoU
    cls_ious = [agg[k]["iou"] for k in agg if "iou" in agg[k]]
    cls_ious = [v for v in cls_ious if not np.isnan(v)]
    if cls_ious:
        agg["__overall__"]["mIoU"] = float(np.mean(cls_ious))
    return agg
