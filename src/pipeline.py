"""전체 추론 파이프라인 (Stage1 → ROI → Patch → Stage2 → Merge)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .patch_utils import (
    compute_roi_bbox,
    crop_patch,
    generate_patch_grid,
)
from .postprocess import (
    confine_to_battery,
    morphology_clean,
    remove_small_components,
    update_max_conf_mask,
)


# Stage1 클래스 인덱스 (학습 시 사용한 순서와 동일해야 함)
STAGE1_CLASSES = {0: "battery_outline", 1: "damaged_large"}
# Stage2
STAGE2_CLASSES = {0: "damaged_small", 1: "pollution"}


class TwoStageBatteryPipeline:
    def __init__(
        self,
        stage1_weights: str,
        stage2_weights: str,
        device: str | int = 0,
        roi_margin: float = 0.1,
        patch_size: int = 320,
        patch_stride: int = 160,
        stage_imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        morph_kernel: int = 3,
        morph_min_area_px: int = 25,
    ):
        self.model_a = YOLO(stage1_weights)
        self.model_b = YOLO(stage2_weights)
        self.device = device
        self.roi_margin = roi_margin
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.imgsz = stage_imgsz
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.morph_kernel = morph_kernel
        self.morph_min_area = morph_min_area_px

    # ──────────────────────────────────────────────
    def _run_stage1(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Returns:
            battery_mask: (H, W) bool
            damaged_large_mask: (H, W) bool
            confs: list of detection confidences (debug용)
        """
        H, W = image.shape[:2]
        battery_mask = np.zeros((H, W), dtype=bool)
        damaged_mask = np.zeros((H, W), dtype=bool)
        confs: List[float] = []

        results = self.model_a.predict(
            image, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            device=self.device, verbose=False,
        )
        if not results or results[0].masks is None:
            return battery_mask, damaged_mask, confs

        r = results[0]
        # ultralytics: r.masks.xy → list of np.ndarray (N, 2) in original image coords
        polys = r.masks.xy
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        scores = r.boxes.conf.cpu().numpy().astype(float)

        for poly, c, s in zip(polys, cls_ids, scores):
            if poly is None or len(poly) < 3:
                continue
            mask_layer = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask_layer, [poly.astype(np.int32)], 1)
            cls_name = STAGE1_CLASSES.get(int(c))
            if cls_name == "battery_outline":
                battery_mask |= mask_layer.astype(bool)
            elif cls_name == "damaged_large":
                damaged_mask |= mask_layer.astype(bool)
            confs.append(float(s))

        return battery_mask, damaged_mask, confs

    # ──────────────────────────────────────────────
    def _run_stage2_on_patches(
        self,
        roi_image: np.ndarray,
        roi_offset: Tuple[int, int],
        full_h: int,
        full_w: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ROI를 patch로 자르고 Model B 돌려 max-conf로 병합.
        Returns: damaged_small_mask (HxW bool), pollution_mask (HxW bool)
        """
        # global accumulator (full image 크기)
        n_classes_b = len(STAGE2_CLASSES)
        # 클래스별 (H, W) conf map → max conf로 픽셀 ownership 결정
        global_conf = np.zeros((full_h, full_w), dtype=np.float32)
        global_label = np.zeros((full_h, full_w), dtype=np.int32)  # 0=BG, 1+=class+1

        roi_h, roi_w = roi_image.shape[:2]
        positions = generate_patch_grid(roi_w, roi_h, self.patch_size, self.patch_stride)

        # batch inference
        BATCH = 8
        for i in range(0, len(positions), BATCH):
            batch_pos = positions[i : i + BATCH]
            patches = [crop_patch(roi_image, x, y, self.patch_size) for (x, y) in batch_pos]
            results = self.model_b.predict(
                patches, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                device=self.device, verbose=False,
            )
            for (px, py), r in zip(batch_pos, results):
                if r.masks is None:
                    continue
                # masks.xy는 patch 입력(=patch_size) 좌표계
                polys = r.masks.xy
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                scores = r.boxes.conf.cpu().numpy().astype(float)
                for poly, c, s in zip(polys, cls_ids, scores):
                    if poly is None or len(poly) < 3:
                        continue
                    pm = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
                    cv2.fillPoly(pm, [poly.astype(np.int32)], 1)
                    pm_bool = pm.astype(bool)
                    # global 좌표: patch local + roi_offset + patch_offset(px, py)
                    ox = roi_offset[0] + px
                    oy = roi_offset[1] + py
                    update_max_conf_mask(
                        global_conf=global_conf,
                        global_label=global_label,
                        patch_mask_bool=pm_bool,
                        patch_conf=float(s),
                        class_id=int(c),
                        offset_xy=(ox, oy),
                    )

        damaged_small_mask = (global_label == (0 + 1))  # damaged_small
        pollution_mask = (global_label == (1 + 1))       # pollution
        return damaged_small_mask, pollution_mask

    # ──────────────────────────────────────────────
    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Args:
            image: BGR uint8 (H, W, 3)
        Returns:
            dict with keys: 'battery_outline', 'damaged', 'pollution' → bool masks
        """
        H, W = image.shape[:2]

        # ── Stage 1
        battery_mask, damaged_large_mask, _ = self._run_stage1(image)

        # ── ROI 추출
        if battery_mask.any():
            ys, xs = np.where(battery_mask)
            x1, y1, x2, y2 = compute_roi_bbox(
                [np.stack([xs, ys], axis=1).astype(np.float32)],
                W, H, margin=self.roi_margin,
            )
        else:
            x1, y1, x2, y2 = 0, 0, W, H
        roi = image[y1:y2, x1:x2]

        # ── Stage 2
        damaged_small_mask, pollution_mask = self._run_stage2_on_patches(
            roi_image=roi, roi_offset=(x1, y1), full_h=H, full_w=W,
        )

        # ── 합치기
        damaged_mask = damaged_large_mask | damaged_small_mask

        masks = {
            "battery_outline": battery_mask,
            "damaged": damaged_mask,
            "pollution": pollution_mask,
        }

        # ── Post-process: confine within battery
        masks = confine_to_battery(masks, battery_mask)

        # ── Morphology + small comp 제거 (defect 클래스만)
        for cls in ("damaged", "pollution"):
            m = masks[cls]
            m = morphology_clean(m, self.morph_kernel)
            m = remove_small_components(m, self.morph_min_area)
            masks[cls] = m

        return masks
