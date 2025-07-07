import os
import shutil
from glob import glob

IMG_SRC = "data/organized_images"
LBL_SRC = "data/yolo_labels"

IMG_DST = "flattened/images"
LBL_DST = "flattened/labels"

os.makedirs(IMG_DST, exist_ok=True)
os.makedirs(LBL_DST, exist_ok=True)

# Copy and rename images
for img_path in glob(f"{IMG_SRC}/batch_*/**/*.*", recursive=True):
    if not img_path.lower().endswith((".jpg", ".jpeg", ".png")): continue
    batch = os.path.basename(os.path.dirname(img_path))
    base = os.path.basename(img_path)
    new_name = f"{batch}_{base}"
    shutil.copy(img_path, os.path.join(IMG_DST, new_name))

# Copy and rename labels
for lbl_path in glob(f"{LBL_SRC}/batch_*/**/*.txt", recursive=True):
    batch = os.path.basename(os.path.dirname(lbl_path))
    base = os.path.basename(lbl_path)
    new_name = f"{batch}_{base}"
    shutil.copy(lbl_path, os.path.join(LBL_DST, new_name))

print("✅ Flattened and renamed image-label pairs.")
