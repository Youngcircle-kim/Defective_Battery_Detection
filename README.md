# Battery Defect Detection (2-Stage YOLOv11-Seg)

원통형 배터리 외관 결함 탐지를 위한 2단계 (Global → Patch) YOLOv11 Segmentation 파이프라인.

## 파이프라인
[1920×1080] 
→ [Stage1: Battery_outline + Damaged_large] (640×640)
→ [ROI + 10% margin]
→ [320×320 Patch, stride 160]
→ [Stage2: Damaged_small + Pollution] (640×640)
→ [Max-conf merge → confine to battery → morphology]
→ [3-class output]

## 사용 순서

```bash
# 0. 설치
pip install -r requirements.txt

# 1. 라벨 분리 (damaged → large/small)
python scripts/01_split_labels.py --config configs/config.yaml

# 2. patch 데이터셋 생성
python scripts/02_generate_patches.py --config configs/config.yaml

# 3. 학습
python scripts/03_train_stage1.py --config configs/config.yaml
python scripts/04_train_stage2.py --config configs/config.yaml

# 4. 추론
python scripts/05_inference.py --image data/Validation/image_data/images/sample.jpg --output runs/inference

# 5. 평가
python scripts/06_evaluate.py --output runs/eval --limit 200
```

## 데이터 구조
data/
├── Training/{image_data/images, label_data/labels}
└── Validation/{image_data/images, label_data/labels}


라벨은 YOLO segmentation format (`class_id x1 y1 ... xn yn`, 정규화).

`source_classes`(config) 를 실제 dataset의 class id에 맞게 수정하세요.

## 주요 설정 포인트 (configs/config.yaml)

| 항목 | 의미 | 기본값 |
|---|---|---|
| `damage_split.threshold_pixels` | damaged → large/small 가르는 픽셀 | 64 |
| `stage2.patch_size` | patch 크기 | 320 |
| `stage2.patch_stride` | patch stride | 160 |
| `stage2.negative_patch_ratio` | positive 대비 negative patch 비율 | 1.0 |
| `inference.patch_merge` | patch overlap 병합 방식 | max_conf |
| `inference.morph_min_area_px` | noise 제거 최소 connected component 크기 | 25 |

## 출력

- `runs/inference/<stem>_visualization.png` — 3-class 오버레이
- `runs/inference/<stem>_polygons.json` — 클래스별 polygon 좌표
- `runs/inference/<stem>_masks/{battery_outline, damaged, pollution}.png` — binary mask
- `runs/eval/metrics_summary.json` — per-class IoU/F1, mIoU, Pixel Accuracy
