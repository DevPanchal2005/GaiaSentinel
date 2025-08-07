import streamlit as st
from PIL import Image
from ultralytics import YOLO

@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")  # yolov8s model you trained

st.title("🔍 Trash Detection")

st.markdown("""
### 💡 Tips for Best Results
- Use clear, well-lit images
- Ensure the trash item is the main subject
- Avoid cluttered backgrounds when possible
- Try different angles for better detection
""")

img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if img:
    image = Image.open(img)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Detecting trash..."):
        model = load_model()
        result = model(image)[0]
        result_image = result.plot()
        st.image(result_image, caption="Detected Objects", use_container_width=True)
else:
    img_col1, img_col2, img_col3 = st.columns(3)

    with img_col1:
        st.image("assests/sample1.jpg", caption="Bottles near Water Body", use_container_width=True)

    with img_col2:
        st.image("assests/sample2.png", caption="Can in vegetation", use_container_width=True)

    with img_col3:
        st.image("assests/sample3.jpg", caption="Cup at a pavement", use_container_width=True)


