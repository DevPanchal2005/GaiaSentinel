import streamlit as st
import pandas as pd

st.set_page_config(page_title="What We Did", layout="wide")
st.title("🧠 Project Walkthrough: Trash Detection with YOLOv8")
st.markdown("---")

st.header("🎯 Problem Statement")
st.markdown("""
In many urban and rural areas, improper waste disposal remains a critical environmental issue.
Manual garbage monitoring is inefficient, subjective, and often delayed. We aimed to build an
automated system that can detect various types of trash (e.g., plastic bottles, cans, wrappers)
in real-time using computer vision.
""")

st.header("🔍 Data Collection & Annotation")
st.markdown("""
- **Sources**: Public datasets, manually captured images.
- **Annotation tool**: Roboflow / CVAT.
- **Format**: YOLO format with bounding boxes.
- **Initial Classes**: 28 trash categories.
- **Split**: 80% training, 20% validation.
""")

st.header("⚙️ Preprocessing & Setup")
st.markdown("""
- **Resized** all images to `640x640`.
- Ensured **normalized bounding box coordinates**.
- Created a `data.yaml` file containing:
  - `path/to/train`, `path/to/val`
  - `names: [list of 14 classes]`
""")

st.header("✂️ Class Reduction for Improved Accuracy")
st.markdown("""
To avoid training instability and boost accuracy, we dropped classes with fewer than **98 annotations**.
This:
- Reduced model confusion on underrepresented classes.
- Allowed the model to focus on more common, better-labeled trash types.
""")

# Table data
before_data = {
    "Super categories": [
        "Plastic bag & wrapper", "Cigarette", "Unlabeled litter", "Bottle", "Bottle cap",
        "Other plastic", "Can", "Carton", "Cup", "Straw", "Paper", "Broken glass",
        "Styrofoam piece", "Pop tab", "Lid", "Plastic container", "Aluminium foil",
        "Plastic utensils", "Rope & strings", "Paper bag", "Scrap metal", "Food waste",
        "Blister pack", "Squeezable tube", "Shoe", "Glass jar", "Plastic glooves", "Battery"
    ],
    "Annotations": [
        850, 667, 517, 439, 289, 273, 273, 251, 192, 161, 148, 138, 112, 99,
        87, 72, 62, 37, 29, 27, 20, 8, 7, 7, 7, 6, 4, 2
    ]
}

after_data = {
    "Super categories": [
        "Plastic bag & wrapper", "Cigarette", "Unlabeled litter", "Bottle", "Bottle cap",
        "Other plastic", "Can", "Carton", "Cup", "Straw", "Paper", "Broken glass",
        "Styrofoam piece", "Pop tab"
    ],
    "Annotations": [
        850, 667, 517, 439, 289, 273, 273, 251, 192, 161, 148, 138, 112, 99
    ]
}

st.subheader("📊 Class Distribution Before and After Filtering")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Before Reduction (28 classes)**")
    df_before = pd.DataFrame(before_data)
    st.dataframe(df_before, use_container_width=True)

with col2:
    st.markdown("**After Reduction (14 classes)**")
    df_after = pd.DataFrame(after_data)
    st.dataframe(df_after, use_container_width=True)

st.header("🧠 Model Training (YOLOv8)")
st.markdown("""
- **Base model**: `yolov8s.pt` (pretrained on COCO).
- **Training**:
    `yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640`
- **Hardware**: NVIDIA T4 GPU (Colab / Paperspace).
- **Epochs**: 100
- **Batch Size**: 16 (depending on memory).
- **Augmentations**: Horizontal flip, scale, HSV shift.
""")

st.header("📈 Evaluation Metrics")
st.markdown("""
- **mAP@0.5**: Used to assess object detection accuracy.
- **Precision / Recall** per class.
- **Confusion matrix** to detect class overlaps.
""")

st.header("📦 Inference & Deployment")
st.markdown("""
- Real-time object detection using webcam or uploaded images.
- Deployed with **Streamlit**, allowing drag-and-drop or camera input.
- Annotated images displayed with class and confidence score.
""")

st.header("💡 Challenges & Lessons")
st.markdown("""
- COCO-pretrained model tried to detect irrelevant classes (e.g., person, chair) → solved by retraining on our dataset.
- Low confidence on minority classes → mitigated by dropping low-count classes.
- Including images without annotations was avoided to prevent degrading performance.
""")

st.header("🚀 What's Next?")
st.markdown("""
- Introduce **instance segmentation**.
- Detect **illegal dumping patterns** using video frames.
- Add **geolocation tagging** for smart waste monitoring.
""")

st.success("This walkthrough reflects the entire model development pipeline.")