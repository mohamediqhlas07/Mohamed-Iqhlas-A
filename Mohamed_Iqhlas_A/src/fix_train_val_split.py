import os
import random
import shutil

TRAIN_DIR = "spectrogram_dataset/train"
VAL_DIR = "spectrogram_dataset/val"
VAL_RATIO = 0.2  # 20%

random.seed(42)

for class_name in os.listdir(TRAIN_DIR):
    class_train_path = os.path.join(TRAIN_DIR, class_name)
    class_val_path = os.path.join(VAL_DIR, class_name)

    if not os.path.isdir(class_train_path):
        continue

    os.makedirs(class_val_path, exist_ok=True)

    images = os.listdir(class_train_path)
    random.shuffle(images)

    val_count = int(len(images) * VAL_RATIO)
    val_images = images[:val_count]

    for img in val_images:
        src = os.path.join(class_train_path, img)
        dst = os.path.join(class_val_path, img)
        shutil.move(src, dst)

    print(f"{class_name}: moved {val_count} images to val")
