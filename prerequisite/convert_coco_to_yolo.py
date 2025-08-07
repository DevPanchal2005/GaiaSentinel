import os, json
from tqdm import tqdm
from PIL import Image

ANNOT_PATH = "data/annotations.json"
IMG_DIR = "data/downloaded_images"
YOLO_LABELS_DIR = "data/yolo_labels"

os.makedirs(YOLO_LABELS_DIR, exist_ok=True)

with open(ANNOT_PATH) as f:
    data = json.load(f)

category_map = {cat['id']: i for i, cat in enumerate(data['categories'])}
img_id_to_size = {img['id']: (img['width'], img['height'], img['file_name']) for img in data['images']}
annos_by_img = {}

for ann in data['annotations']:
    annos_by_img.setdefault(ann['image_id'], []).append(ann)

for img_id, anns in tqdm(annos_by_img.items()):
    w, h, fname = img_id_to_size[img_id]
    out_path = os.path.join(YOLO_LABELS_DIR, os.path.splitext(fname)[0] + ".txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for ann in anns:
            x, y, bw, bh = ann['bbox']
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            bw /= w
            bh /= h
            cls = category_map[ann['category_id']]
            f.write(f"{cls} {xc} {yc} {bw} {bh}\n")
