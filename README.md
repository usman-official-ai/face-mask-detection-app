# Real-Time Face Mask Detection Using Deep Learning

## Project Overview
This project detects whether a person is wearing a face mask using deep learning and computer vision. It supports image uploads and real-time webcam detection.

## Live Demo
🔗 **Streamlit App:** https://face-mask-detection-app-fj4rtaayswd6dywttpcymh.streamlit.app/  

## Features
- Deep learning-based face mask detection
- Real-time webcam detection using OpenCV
- Image upload prediction
- Confidence score display
- REST API using FastAPI

## Tech Stack
- Python 3.12+
- TensorFlow / Keras
- OpenCV
- FastAPI
- Streamlit

## Models Trained
1. Custom CNN - 84.18% accuracy
2. MobileNetV2 - 98.27% accuracy ✅ (Best)
3. ResNet50 - Underfit (skipped)

## Results
| Model | Validation Accuracy |
|-------|---------------------|
| Custom CNN | 84.18% |
| **MobileNetV2** | **98.27%** |

## Installation
```bash
pip install -r requirements.txt


How to Run
bash
# Streamlit app
streamlit run app.py

# Real-time webcam detection
python src/detection.py

# FastAPI server
python api/app.py

GitHub Repository
🔗 https://github.com/SoftCr8ors/Deep-Learning-Face-Mask-Detection.git

Author
Muhammad Usman

Organization
SoftCr8ors
