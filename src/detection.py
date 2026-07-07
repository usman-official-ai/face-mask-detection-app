"""
Real-Time Face Mask Detection using OpenCV
Author: Muhammad Usman
Company: SoftCr8ors
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

class FaceMaskDetector:
    def __init__(self, model_path='models/face_mask_model.h5'):
        self.model = load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.img_size = 128
        
    def preprocess_face(self, face):
        face = cv2.resize(face, (self.img_size, self.img_size))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)
        return face
    
    def predict_face(self, face):
        processed_face = self.preprocess_face(face)
        prediction = self.model.predict(processed_face, verbose=0)
        pred_value = float(prediction[0][0])  # Convert to float
        
        # Labels swapped - fixed
        if pred_value > 0.5:
            label = "Without Mask"
            confidence = pred_value * 100
        else:
            label = "With Mask"
            confidence = (1 - pred_value) * 100
        
        return label, confidence
    
    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            label, confidence = self.predict_face(face_roi)
            
            color = (0, 255, 0) if label == 'With Mask' else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = f"{label}: {confidence:.2f}%"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def run_webcam(self):
        cap = cv2.VideoCapture(0)
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.detect(frame)
            cv2.imshow('Face Mask Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FaceMaskDetector()
    detector.run_webcam()
    