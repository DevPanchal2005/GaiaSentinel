# 🌍 GaiaSentinel — Watching Earth, Detecting Waste

A real-world computer vision application for automated trash detection and classification using YOLOv8 and the [TACO dataset](http://tacodataset.org/). Built with Streamlit and deployable on Streamlit Community Cloud.

---

## 🚀 Live Demo

> Upload any image and GaiaSentinel will detect and classify waste items in real time.

Website Under Construction 🚧
---

## 🧠 What It Does

GaiaSentinel uses a fine-tuned **YOLOv8s** model trained on the TACO (Trash Annotations in Context) dataset to identify **14 categories of litter** in the wild, including plastic bags, bottles, cans, cigarettes, and more.

| Capability | Details |
|---|---|
| 🔍 Detection | Bounding box detection with confidence scores |
| 🏷️ Classification | 14 trash categories |
| 📸 Input | Image upload (JPG, PNG, JPEG) |
| ⚡ Speed | CPU-optimized for cloud deployment |

---

## 🗂️ Project Structure

```
GaiaSentinel/
├── app.py                        # Main Streamlit entry point & navigation
├── data.yaml                     # YOLOv8 dataset config (14 classes)
├── train.py                      # Model training script
├── requirements.txt              # Python dependencies
├── assets/                       # Sample images & logos
├── pages/
│   ├── Welcome.py                # Home dashboard with category stats
│   ├── Detect_Trash.py           # Core detection interface
│   └── What_We_Did.py            # Project walkthrough & methodology
├── prerequisite/
│   ├── download_images.py        # Download TACO images from Flickr URLs
│   ├── rename_images.py          # Organize images by batch
│   ├── convert_coco_to_yolo.py   # COCO JSON → YOLO format conversion
│   ├── cat_to_supCat.ipynb       # Remap 60 subcategories → 28 supercategories
│   ├── reduce_Cat.ipynb          # Filter down to 14 final classes
│   ├── flatten_data.py           # Flatten batch folders into flat structure
│   ├── split_yolo_dataset.py     # 80/20 train/val split
│   └── visualize.py              # Visualize annotations on images
└── .streamlit/
    └── config.toml               # Theme & logger configuration
```

---

## 🏷️ Detected Trash Categories

| ID | Category | ID | Category |
|---|---|---|---|
| 0 | Plastic bag & wrapper | 7 | Carton |
| 1 | Cigarette | 8 | Cup |
| 2 | Unlabeled litter | 9 | Straw |
| 3 | Bottle | 10 | Paper |
| 4 | Bottle cap | 11 | Broken glass |
| 5 | Other plastic | 12 | Styrofoam piece |
| 6 | Can | 13 | Pop tab |

---

## 🔧 Data Pipeline

The TACO dataset required significant preprocessing before training. Here's how raw data became a trained model:

```
TACO JSON Annotations
        │
        ▼
1. download_images.py       → Download images from Flickr URLs
2. rename_images.py         → Organize into batch_N/image.jpg structure
3. convert_coco_to_yolo.py  → Convert COCO bbox format to YOLO format
4. cat_to_supCat.ipynb      → Remap 60 fine-grained classes → 28 supercategories
5. reduce_Cat.ipynb         → Keep 14 classes with ≥ 99 annotations
6. flatten_data.py          → Flatten batches into a single image/label directory
7. split_yolo_dataset.py    → 80% train / 20% val split
        │
        ▼
    Training-ready YOLO dataset
```

**Key decisions:**
- Started with **60 subcategories**, consolidated to **28 supercategories**, then filtered to **14 final classes** by dropping anything with fewer than 99 annotations — reducing model confusion and improving precision on well-represented classes.
- 1,500 images processed, 4,784 annotations remapped (98.5% required class ID changes).

---

## 🤖 Model Training

```bash
yolo detect train \
  model=yolov8s.pt \
  data=data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0
```

- **Base model:** `yolov8s.pt` (pretrained on COCO)
- **Hardware:** NVIDIA T4 GPU (Google Colab / Paperspace)
- **Augmentations:** Horizontal flip, HSV shift, scale jitter
- **Output:** `runs/detect/train/weights/best.pt`

---

## 🖥️ Running Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/GaiaSentinel.git
cd GaiaSentinel
```

### 2. Install dependencies

```bash
pip install streamlit ultralytics opencv-python-headless Pillow
```

### 3. Add your trained model

Place your trained `best.pt` file in the project root.

### 4. Run the app

```bash
streamlit run app.py
```

---


## 📈 Results

| Metric | Value |
|---|---|
| Training classes | 14 |
| Training images | ~1,200 |
| Validation images | ~300 |
| Evaluation metric | mAP@0.5 |

---

## 🚀 What's Next

- **Instance segmentation** using YOLOv8-seg for pixel-level masks
- **Geolocation tagging** to map trash density by area
- **Recyclability classification** — mapping detections to recyclable / non-recyclable / hazardous
- **Video & drone support** for monitoring public spaces over time

---

## 📦 Dataset

This project uses the **TACO Dataset** — Trash Annotations in Context.

> Proença, P.F. and Simões, P., 2020. TACO: Trash annotations in context for litter detection. arXiv preprint arXiv:2003.06975.

[http://tacodataset.org](http://tacodataset.org) · [GitHub](https://github.com/pedropro/TACO)

---

## 📄 License

This project is for educational and research purposes. TACO dataset usage is subject to its own license terms.
