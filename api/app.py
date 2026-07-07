import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Face Mask Detection", layout="centered")

st.title("😷 Face Mask Detection")

@st.cache_resource
def load_model_cached():
    return load_model('face_mask_model_finetuned.h5')

model = load_model_cached()
IMG_SIZE = 128

# ========== TABS ==========
tab1, tab2 = st.tabs(["📸 Upload Image", "🎥 Webcam"])

# ========== TAB 1: IMAGE UPLOAD ==========
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
            st.error(f"❌ Without Mask ({pred*100:.2f}%)")
        else:
            st.success(f"✅ With Mask ({(1-pred)*100:.2f}%)")

# ========== TAB 2: WEBCAM ==========
with tab2:
    st.info("Webcam works only on local machine. Click 'Start Webcam' to run.")
    
    if st.button("🎥 Start Webcam"):
        st.warning("Webcam will open in a separate OpenCV window. Press 'q' to quit.")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam.")
        else:
            st.success("Webcam running... Press 'q' to quit.")
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                    face_norm = face_resized / 255.0
                    face_input = np.expand_dims(face_norm, axis=0)
                    
                    pred = model.predict(face_input, verbose=0)[0][0]
                    
                    if pred > 0.5:
                        label = "Without Mask"
                        color = (0, 0, 255)
                    else:
                        label = "With Mask"
                        color = (0, 255, 0)
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{label}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.imshow("Face Mask Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            st.info("Webcam stopped.")