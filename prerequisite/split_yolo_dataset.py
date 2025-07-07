import os
import shutil
import random
from glob import glob

IMG_DIR = "data/images"
LBL_DIR = "data/labels"
OUT_DIR = "data"
TRAIN_RATIO = 0.8

os.makedirs(f"{OUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/images/val", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/val", exist_ok=True)

all_images = glob(os.path.join(IMG_DIR, "*.jpg")) + glob(os.path.join(IMG_DIR, "*.jpeg")) + glob(os.path.join(IMG_DIR, "*.png"))
random.shuffle(all_images)

split_index = int(len(all_images) * TRAIN_RATIO)
train_imgs = all_images[:split_index]
val_imgs = all_images[split_index:]

def copy_pairs(image_list, subset):
    for img_path in image_list:
        base = os.path.basename(img_path)
        lbl_path = os.path.join(LBL_DIR, os.path.splitext(base)[0] + ".txt")
        if not os.path.exists(lbl_path): continue  # skip unpaired

        shutil.copy(img_path, os.path.join(OUT_DIR, "images", subset, base))
        shutil.copy(lbl_path, os.path.join(OUT_DIR, "labels", subset, os.path.basename(lbl_path)))

copy_pairs(train_imgs, "train")
copy_pairs(val_imgs, "val")

print(f"✅ Done. {len(train_imgs)} train, {len(val_imgs)} val samples.")
