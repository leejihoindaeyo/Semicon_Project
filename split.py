# dataset_split.py

import os
import shutil
import random

def split_dataset(source_dir, target_dir, split_ratio=0.8):
    classes = os.listdir(source_dir)
    print(f"Classes: {classes}")

    for cls in classes:
        src_cls_dir = os.path.join(source_dir, cls)
        if not os.path.isdir(src_cls_dir):
            continue

        files = [f for f in os.listdir(src_cls_dir) if f.endswith('.mp4')]
        random.shuffle(files)

        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        # Target dirs
        train_cls_dir = os.path.join(target_dir, 'train', cls)
        val_cls_dir = os.path.join(target_dir, 'val', cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir, exist_ok=True)

        # Copy files
        for f in train_files:
            shutil.copy(os.path.join(src_cls_dir, f), os.path.join(train_cls_dir, f))

        for f in val_files:
            shutil.copy(os.path.join(src_cls_dir, f), os.path.join(val_cls_dir, f))

        print(f"[✅] {cls}: train={len(train_files)} files, val={len(val_files)} files")

# 사용 예시
source_dataset = '/home/yunddu/Semicon_Project/DataSet'  # 원본 Dataset 경로 (daily/falldown 폴더만 있음)
target_dataset = '/home/yunddu/Semicon_Project/Dataset'       # 새로 split할 Dataset 경로 (train/val 자동 생성됨)

split_dataset(source_dataset, target_dataset, split_ratio=0.8)
