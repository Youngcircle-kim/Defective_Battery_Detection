# scripts/00_prepare_dataset.py
"""
다운로드한 AI Hub zip들의 압축해제 결과를 우리 프로젝트 구조로 재배치하고,
Training의 11%를 Test로 분리.

사용법:
    1. raw_extracted/ 아래에 압축이 풀려있어야 함
    2. python scripts/00_prepare_dataset.py --raw raw_extracted --out data --test-ratio 0.11
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

from tqdm import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def find_all_files(root: Path, exts: set[str]) -> list[Path]:
    """root 아래 모든 하위 디렉토리에서 확장자 매칭 파일 재귀 검색."""
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def safe_move(src: Path, dst: Path):
    """dst.parent 만들고 이동. 동일 이름 있으면 skip."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.move(str(src), str(dst))


def consolidate_training_images(raw_root: Path, target: Path):
    """
    raw_root/Training/images_tmp_{1..4}/.../RGB_*.png  →  target/*.png
    여러 zip이 분리돼 풀린 걸 하나의 폴더로 합침.
    """
    target.mkdir(parents=True, exist_ok=True)
    sources = sorted(raw_root.glob("Training/images_tmp_*"))
    print(f"[Training images] {len(sources)}개 소스 폴더 발견")
    for src_root in sources:
        files = find_all_files(src_root, IMG_EXTS)
        print(f"  - {src_root.name}: {len(files)}장 → 이동")
        for f in tqdm(files, desc=src_root.name):
            safe_move(f, target / f.name)


def consolidate_simple(raw_dir: Path, target: Path, exts: set[str], label: str):
    """단일 소스 디렉토리 → 하나의 평탄한 디렉토리로 합치기."""
    target.mkdir(parents=True, exist_ok=True)
    files = find_all_files(raw_dir, exts)
    print(f"[{label}] {len(files)}개 파일 → {target}")
    for f in tqdm(files, desc=label):
        safe_move(f, target / f.name)


def split_train_into_train_test(
    train_img_dir: Path,
    train_lbl_dir: Path,
    test_img_dir: Path,
    test_lbl_dir: Path,
    test_ratio: float,
    seed: int = 42,
):
    """
    Training 이미지의 일정 비율을 Test로 분리.
    - 이미지 stem 기준으로 sampling
    - 라벨도 같이 이동 (.json)
    """
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(p for p in train_img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    n_total = len(img_files)
    n_test = int(round(n_total * test_ratio))
    print(f"[Test split] Train {n_total}장 중 {n_test}장 ({test_ratio*100:.1f}%)을 Test로 이동")

    rng = random.Random(seed)
    rng.shuffle(img_files)
    test_files = img_files[:n_test]

    n_lbl_moved = 0
    n_lbl_missing = 0
    for img_path in tqdm(test_files, desc="train→test"):
        # 이미지 이동
        shutil.move(str(img_path), str(test_img_dir / img_path.name))
        # 라벨 이동
        lbl_src = train_lbl_dir / f"{img_path.stem}.json"
        if lbl_src.exists():
            shutil.move(str(lbl_src), str(test_lbl_dir / lbl_src.name))
            n_lbl_moved += 1
        else:
            n_lbl_missing += 1

    print(f"  라벨 이동: {n_lbl_moved}, 라벨 누락: {n_lbl_missing}")


def verify_pairing(img_dir: Path, lbl_dir: Path, name: str):
    """이미지 ↔ 라벨 stem 매칭 검증."""
    img_stems = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    lbl_stems = {p.stem for p in lbl_dir.iterdir() if p.suffix.lower() == ".json"}
    matched = img_stems & lbl_stems
    img_only = img_stems - lbl_stems
    lbl_only = lbl_stems - img_stems
    print(f"\n[{name}] 매칭 검증")
    print(f"  이미지: {len(img_stems)}, 라벨: {len(lbl_stems)}, 매칭: {len(matched)}")
    if img_only:
        print(f"  ⚠ 라벨 없는 이미지: {len(img_only)}개 (예: {next(iter(img_only))})")
    if lbl_only:
        print(f"  ⚠ 이미지 없는 라벨: {len(lbl_only)}개 (예: {next(iter(lbl_only))})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default="raw_extracted",
                    help="압축 해제된 임시 디렉토리")
    ap.add_argument("--out", type=str, default="Defective_Battery_Detection/data",
                    help="최종 데이터 디렉토리")
    ap.add_argument("--test-ratio", type=float, default=0.11,
                    help="Training에서 Test로 떼낼 비율 (기본 0.11 → 8:1:1)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)

    train_img = out / "Training/image_data/images"
    train_lbl = out / "Training/label_data/labels"
    val_img   = out / "Validation/image_data/images"
    val_lbl   = out / "Validation/label_data/labels"
    test_img  = out / "Test/image_data/images"
    test_lbl  = out / "Test/label_data/labels"

    # ── 1. Training 이미지 통합 (4개 zip 결과 → 1개 폴더)
    print("\n=== [1/4] Training 이미지 통합 ===")
    consolidate_training_images(raw, train_img)

    # ── 2. Training 라벨, Validation 이미지/라벨 통합
    print("\n=== [2/4] Training 라벨 / Validation 통합 ===")
    consolidate_simple(raw / "Training/labels_tmp", train_lbl, {".json"}, "Train labels")
    consolidate_simple(raw / "Validation/images_tmp", val_img, IMG_EXTS, "Val images")
    consolidate_simple(raw / "Validation/labels_tmp", val_lbl, {".json"}, "Val labels")

    # ── 3. Train/Val 매칭 검증 (Test 분리 전)
    print("\n=== [3/4] 매칭 검증 (Test 분리 전) ===")
    verify_pairing(train_img, train_lbl, "Training (전체)")
    verify_pairing(val_img, val_lbl, "Validation")

    # ── 4. Train의 일부 → Test로 이동
    print("\n=== [4/4] Train → Test 분리 ===")
    split_train_into_train_test(
        train_img, train_lbl, test_img, test_lbl,
        test_ratio=args.test_ratio, seed=args.seed,
    )

    # ── 최종 검증
    print("\n=== 최종 매칭 검증 ===")
    verify_pairing(train_img, train_lbl, "Training (분리 후)")
    verify_pairing(val_img, val_lbl, "Validation")
    verify_pairing(test_img, test_lbl, "Test")

    print("\n✅ 완료. 다음 단계:")
    print("    python scripts/01_split_labels.py --config configs/config.yaml")


if __name__ == "__main__":
    main()