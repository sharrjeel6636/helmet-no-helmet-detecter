import streamlit as st
import cv2
import PIL.Image
import tempfile
import os
from ultralytics import YOLO
import pandas as pd

# --- Page Config ---
st.set_page_config(
    page_title="Safety Helmet Detection",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        text-align: center;
        margin-bottom: 2rem;
    }
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 18px;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, max_value=1.0, value=0.25, step=0.05,
    help="Minimum confidence score for detections to be shown."
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold", 
    min_value=0.0, max_value=1.0, value=0.45, step=0.05,
    help="Intersection Over Union threshold for NMS."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Information")
st.sidebar.info("Model: standard YOLOv8n (Custom Trained)\nClasses: Head, Helmet, Person")

# --- Model Loading ---
@st.cache_resource
def load_model(model_path):
    """Loads the YOLO model and caches it to prevent reloading."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

MODEL_PATH = "best (3).pt"
model = load_model(MODEL_PATH)

# --- Main Content ---
st.markdown('<div class="main-header">👷 Safety Helmet Detection Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload an image to detect safety compliance</div>', unsafe_allow_html=True)

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📸 Original Image")
        image = PIL.Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    if model:
        # Inference button
        if st.sidebar.button("🔍 Run Detection"):
            with st.spinner("Analyzing image..."):
                try:
                    # Run inference
                    results = model.predict(image, conf=confidence_threshold, iou=iou_threshold)
                    
                    # Process results
                    result = results[0]
                    res_plotted = result.plot()
                    res_image = PIL.Image.fromarray(res_plotted[..., ::-1]) # Convert BGR to RGB

                    # Display Result
                    with col2:
                        st.markdown("### ✅ Detected Output")
                        st.image(res_image, use_container_width=True)

                    # Metrics Section
                    st.markdown("---")
                    st.markdown("### 📊 Detection Statistics")
                    
                    # Count classes
                    class_counts = {}
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        cls_name = model.names[cls]
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    
                    if class_counts:
                        # Create metrics row
                        cols = st.columns(len(class_counts))
                        for idx, (cls_name, count) in enumerate(class_counts.items()):
                            with cols[idx]:
                                st.metric(label=cls_name.upper(), value=count)
                                
                        # Detailed Box Data
                        with st.expander("See Detailed Detection Data"):
                            box_data = []
                            for box in result.boxes:
                                box_data.append({
                                    "Class": model.names[int(box.cls[0])],
                                    "Confidence": f"{float(box.conf[0]):.2f}",
                                    "Coordinates": f"{[round(x,1) for x in box.xyxy[0].tolist()]}"
                                })
                            df = pd.DataFrame(box_data)
                            st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("No objects detected with current settings.")
                        
                except Exception as e:
                    st.error(f"Error during detection: {e}")
    else:
        st.error("Model failed to load. Please check the model file path.")

else:
    # Landing page info
    st.info("👆 Please upload an image to get started.")
