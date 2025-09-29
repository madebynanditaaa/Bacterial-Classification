# src/dataset_split.py
"""
Create stratified train/val/test splits by copying images from Dataset/ and Dataset_Augmented/
into data/splits/{train,val,test}/{class_name}/

Run:
    python src/dataset_split.py
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict

from config import RAW_DATASET_DIR, AUG_DATASET_DIR, OUTPUT_SPLIT_DIR, TRAIN_SPLIT, VAL_SPLIT, SEED

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def gather_images_from_folder(root_dir):
    items = []
    if not os.path.exists(root_dir):
        return []
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if Path(fname).suffix.lower() in IMG_EXTS:
                items.append((os.path.join(class_dir, fname), class_name))
    return items


def collect_all_images():
    all_items = []
    for d in [RAW_DATASET_DIR, AUG_DATASET_DIR]:
        all_items.extend(gather_images_from_folder(d))
    return all_items


def sanitize_class_name(name: str) -> str:
    # convert spaces and unusual chars to underscores
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name.replace(" ", "_"))


def make_dirs_for_split(split_name, classes):
    base = os.path.join(OUTPUT_SPLIT_DIR, split_name)
    os.makedirs(base, exist_ok=True)
    for cl in classes:
        os.makedirs(os.path.join(base, cl), exist_ok=True)


def save_files_to_split(items, split_name):
    """
    items: list of tuples (src_path, class_name)
    split_name: 'train'|'val'|'test'
    """
    for src, cls in items:
        cls_s = sanitize_class_name(cls)
        dest_dir = os.path.join(OUTPUT_SPLIT_DIR, split_name, cls_s)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(src))
        try:
            shutil.copy2(src, dest)
        except Exception as e:
            print(f"Failed to copy {src} -> {dest}: {e}")


def split_and_save():
    items = collect_all_images()
    if not items:
        raise SystemExit(f"No images found in {RAW_DATASET_DIR} or {AUG_DATASET_DIR}")

    paths = [p for p, c in items]
    labels = [c for p, c in items]

    # Train vs temp
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=1 - TRAIN_SPLIT, stratify=labels, random_state=SEED
    )
    # val vs test from temp (relative)
    relative_val_size = VAL_SPLIT / (1 - TRAIN_SPLIT)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=1 - relative_val_size, stratify=temp_labels, random_state=SEED
    )

    train_items = list(zip(train_paths, train_labels))
    val_items = list(zip(val_paths, val_labels))
    test_items = list(zip(test_paths, test_labels))

    classes = sorted(set(labels))
    print("Classes found:", classes)
    make_dirs_for_split("train", [sanitize_class_name(c) for c in classes])
    make_dirs_for_split("val", [sanitize_class_name(c) for c in classes])
    make_dirs_for_split("test", [sanitize_class_name(c) for c in classes])

    print(f"Saving {len(train_items)} train, {len(val_items)} val, {len(test_items)} test images...")
    save_files_to_split(train_items, "train")
    save_files_to_split(val_items, "val")
    save_files_to_split(test_items, "test")
    print("Done.")


if __name__ == "__main__":
    split_and_save()
