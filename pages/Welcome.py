import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px

st.set_page_config(page_title="GaiaSentinel", page_icon=":guardsman:", layout="wide")

# Page header
st.title("🗑️ GaiaSentinel - Watching Earth, Detecting Waste")

# Hero section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## About This Application
    
    GaiaSentinel uses advanced computer vision and machine learning 
    to automatically identify and classify different types of waste materials. 
    This helps in:
    
    - **♻️ Waste Sorting**: Automatically categorize trash for proper recycling
    - **🌍 Environmental Impact**: Reduce contamination in recycling streams
    - **📊 Data Analytics**: Track waste patterns and composition
    - **🎯 Education**: Learn about different waste categories
    """)

with col2:
    # You can add an image here if available
    st.image("assests/taco_logo.png", use_container_width=True)

# Features section
st.markdown("## 🚀 Key Features")

feature_cols = st.columns(3)

with feature_cols[0]:
    with st.container(border=True):
        st.markdown("""
        ### 📸 Image Detection
        - Upload images for instant classification
        - Real-time processing
        - High accuracy detection
        - Multiple format support
        """)

with feature_cols[1]:
    with st.container(border=True):
        st.markdown("""
        ### 📊 Analytics Dashboard
        - Detailed classification reports
        - Visual data insights
        - Performance metrics
        - Historical tracking
        """)

with feature_cols[2]:
    with st.container(border=True):
        st.markdown("""
        ### 🎯 14 Waste Categories
        - Plastic bags & wrappers
        - Cigarettes
        - Bottles & caps
        - Cans & containers and 10 more categories 
        """)

# How it works section
st.markdown("---")
st.markdown("## 🔧 How It Works")

process_cols = st.columns(4)

with process_cols[0]:
    st.markdown("""
    <div style="text-align: center;">
        <div style="font-size: 40px;">📤</div>
        <h4>1. Upload</h4>
        <p>Upload your trash image</p>
    </div>
    """, unsafe_allow_html=True)

with process_cols[1]:
    st.markdown("""
    <div style="text-align: center;">
        <div style="font-size: 40px;">🔍</div>
        <h4>2. Analyze</h4>
        <p>Model processes the image</p>
    </div>
    """, unsafe_allow_html=True)

with process_cols[2]:
    st.markdown("""
    <div style="text-align: center;">
        <div style="font-size: 40px;">🏷️</div>
        <h4>3. Classify</h4>
        <p>Identify waste category</p>
    </div>
    """, unsafe_allow_html=True)

with process_cols[3]:
    st.markdown("""
    <div style="text-align: center;">
        <div style="font-size: 40px;">📋</div>
        <h4>4. Report</h4>
        <p>Get detailed results</p>
    </div>
    """, unsafe_allow_html=True)

# Getting started section
st.markdown("---")
st.markdown("## 🎯 Getting Started")

st.markdown("""
Ready to start detecting trash? Here's how to begin:

1. **Navigate to Detection**: Use the sidebar to access the detection page
2. **Upload an Image**: Choose a clear image of waste items
3. **Get Results**: View the AI's classification and confidence scores
4. **Explore Analytics**: Check out detailed insights and reports
""")

# Statistics section (sample data)
st.markdown("---")
st.markdown("## 📈 Detection Categories")

# Sample statistics for the 14 categories
categories_data = {
    'Category': [
        'Plastic bag & wrapper', 'Cigarette', 'Unlabeled litter', 'Bottle',
        'Bottle cap', 'Other plastic', 'Can', 'Carton', 'Cup', 'Straw',
        'Paper', 'Broken glass', 'Styrofoam piece', 'Pop tab'
    ],
    'Annotations': [850, 667, 517, 439, 289, 273, 273, 251, 192, 161, 148, 138, 112, 99],
    'Category ID': list(range(14))
}

df = pd.DataFrame(categories_data)

# Create a bar chart
fig = px.bar(
    df, 
    x='Category', 
    y='Annotations',
    title='Training Data Distribution by Category',
    color='Annotations',
    color_continuous_scale='viridis'
)
fig.update_xaxes(tickangle=45)
fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)



