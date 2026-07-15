"""
Full data preparation pipeline.
Run this once after downloading the raw datasets.
"""
import os, shutil, hashlib
from pathlib import Path
from collections import Counter
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import imagehash

random.seed(42); np.random.seed(42)

PROJECT_ROOT = "/kaggle/working/waste_classificator"
SOURCE_DATASETS = [
    "/kaggle/input/datasets/asdasdasasdas/garbage-classification/Garbage classification/Garbage classification",
    "/kaggle/input/datasets/feyzazkefe/trashnet/dataset-resized",
]
MERGED = Path(f"{PROJECT_ROOT}/data/raw/merged")
DEST   = Path(f"{PROJECT_ROOT}/data/processed")


def merge_sources():
    MERGED.mkdir(parents=True, exist_ok=True)
    for src_path in SOURCE_DATASETS:
        src = Path(src_path)
        if not src.exists(): continue
        prefix = src_path.split('/')[-2]
        for cls_folder in src.iterdir():
            if not cls_folder.is_dir(): continue
            dest_cls = MERGED / cls_folder.name.lower()
            dest_cls.mkdir(exist_ok=True)
            for img in cls_folder.iterdir():
                if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    shutil.copy(img, dest_cls / f"{prefix}_{img.name}")


def deduplicate():
    hashes = {}
    dups = []
    for cls in MERGED.iterdir():
        if not cls.is_dir(): continue
        for img in cls.glob("*.*"):
            try:
                h = imagehash.phash(Image.open(img).convert("RGB"), hash_size=8)
                if h in hashes: dups.append((img, hashes[h]))
                else: hashes[h] = img
            except Exception: pass
    for dup, orig in dups: dup.unlink()
    return len(dups)


def clean_corrupt():
    bad = []
    for img in MERGED.rglob("*.*"):
        try:
            if img.stat().st_size < 1024: bad.append(img); continue
            with Image.open(img) as im:
                w, h = im.size
                if w < 32 or h < 32 or max(w, h) / min(w, h) > 10: bad.append(img)
        except Exception: bad.append(img)
    for b in bad: b.unlink()
    return len(bad)


def stratified_split():
    files, labels = [], []
    for cls in sorted(MERGED.iterdir()):
        if not cls.is_dir(): continue
        for img in cls.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                files.append(img); labels.append(cls.name)
    train_f, temp_f, train_l, temp_l = train_test_split(
        files, labels, test_size=0.30, stratify=labels, random_state=42)
    val_f, test_f, val_l, test_l = train_test_split(
        temp_f, temp_l, test_size=0.50, stratify=temp_l, random_state=42)
    for split, fs, ls in [("train", train_f, train_l), ("val", val_f, val_l), ("test", test_f, test_l)]:
        d = DEST / split
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True)
        for f, l in zip(fs, ls):
            sd = d / l; sd.mkdir(exist_ok=True)
            shutil.copy(f, sd / f.name)


if __name__ == "__main__":
    print("→ Merging sources...");   merge_sources()
    print("→ Deduplicating...");      n = deduplicate();         print(f"  removed {n} dups")
    print("→ Cleaning corrupt...");   n = clean_corrupt();       print(f"  removed {n} bad files")
    print("→ Stratified split...");   stratified_split()
    print("✓ Done. Final counts:")
    for split in ["train", "val", "test"]:
        c = sum(1 for _ in (DEST / split).rglob("*.*"))
        print(f"  {split}: {c}")
