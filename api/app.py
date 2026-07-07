import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Face Mask Detection", layout="centered")

st.title("😷 Face Mask Detection")
st.write("Upload an image or use webcam for real-time detection")

@st.cache_resource
def load_model_cached():
    return load_model('models/face_mask_model.h5')

model = load_model_cached()
IMG_SIZE = 128

# Tab selection
tab1, tab2 = st.tabs(["📸 Image Upload", "🎥 Webcam"])

# ==================== TAB 1: Image Upload ====================
with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img = np.array(image)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        input_data = np.expand_dims(normalized, axis=0)

        pred = model.predict(input_data, verbose=0)[0][0]

        if pred > 0.5:
            label = "Without Mask"
            confidence = pred * 100
        else:
            label = "With Mask"
            confidence = (1 - pred) * 100

        if label == "With Mask":
            st.success(f"✅ {label} ({confidence:.2f}%)")
        else:
            st.error(f"❌ {label} ({confidence:.2f}%)")

# ==================== TAB 2: Webcam ====================
with tab2:
    st.write("Click 'Start Webcam' to begin real-time detection.")
    
    run_webcam = st.button("🎥 Start Webcam", type="primary")
    
    if run_webcam:
        st.info("Webcam started. Press 'q' in the webcam window to quit.")
        
        # Run the webcam detection script
        import subprocess
        import sys
        subprocess.run([sys.executable, "src/detection.py"])