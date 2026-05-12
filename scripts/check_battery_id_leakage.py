from pathlib import Path
import re

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def extract_id(path: Path):
    stem = path.stem
    return re.split(r"[_\-]", stem)[0]

def collect_ids(image_dir):
    ids = set()
    for p in Path(image_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            ids.add(extract_id(p))
    return ids

train_ids = collect_ids("data_stage1/images/train")
val_ids = collect_ids("data_stage1/images/val")

overlap = train_ids & val_ids

print("train IDs:", len(train_ids))
print("val IDs:", len(val_ids))
print("overlap IDs:", len(overlap))
print("examples:", list(sorted(overlap))[:20])