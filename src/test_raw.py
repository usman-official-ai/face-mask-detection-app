"""
Test raw prediction value
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load model
model = load_model('models/face_mask_model.h5')
IMG_SIZE = 128

# Ask for image path
img_path = input("Enter image path: ")

# Check if file exists
if not os.path.exists(img_path):
    print(f"Error: Image not found at {img_path}")
    exit()

# Read and preprocess image
img = cv2.imread(img_path)
if img is None:
    print("Error: Could not read image. Please check the file.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
normalized = resized / 255.0
input_data = np.expand_dims(normalized, axis=0)

# Predict
pred = model.predict(input_data, verbose=0)[0][0]
print(f"Raw prediction score: {pred:.4f}")

# Labels are swapped - fixed
if pred > 0.5:
    print("Without Mask")  # Swapped
else:
    print("With Mask")     # Swapped