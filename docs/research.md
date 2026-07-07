# Face Mask Detection - Research Document

## 1. Problem Statement
Develop a deep learning system to detect whether a person is wearing a face mask using computer vision techniques. The system should work on both images and real-time video.

## 2. Computer Vision Fundamentals
Computer Vision is a field of AI that enables computers to interpret and understand visual information from the world. Key concepts include:
- Image representation (pixels, RGB channels)
- Feature extraction
- Object detection
- Image classification

## 3. CNN Architecture Overview
Convolutional Neural Networks (CNNs) are specialized neural networks for processing grid-like data (images).

### Key Components:
- **Convolutional Layers**: Apply filters to extract features (edges, textures, shapes)
- **Pooling Layers**: Reduce spatial dimensions (MaxPooling, AveragePooling)
- **Fully Connected Layers**: Perform classification based on extracted features
- **Activation Functions**: ReLU, Sigmoid, Softmax

### CNN Architecture Flow:
  
  Input Image → Conv → ReLU → Pool → Conv → ReLU → Pool → FC → Output  
    
    
## 4. Transfer Learning Models

### MobileNetV2
- Lightweight model designed for mobile and embedded vision applications
- Uses depthwise separable convolutions
- Parameters: ~3.5 million
- Good balance between accuracy and computational cost

### ResNet50
- Deep residual network with 50 layers
- Uses skip connections to solve vanishing gradient problem
- Parameters: ~25 million
- Higher accuracy but computationally expensive

## 5. Dataset Information
- **Source**: Kaggle Face Mask Detection Dataset
- **Total Images**: 7,553
- **Classes**: 2 (With Mask, Without Mask)
- **Distribution**:
  - With Mask: 3,725 images
  - Without Mask: 3,828 images

## 6. Data Preprocessing Steps
1. Resize images to 224x224 pixels
2. Normalize pixel values (0-1 range)
3. Apply data augmentation (rotation, zoom, flip, shift)
4. Split data: 80% Training, 10% Validation, 10% Testing

## 7. Evaluation Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of Precision and Recall
- **Confusion Matrix**: Visual representation of predictions vs actual

## 8. Expected Outcome
- A trained deep learning model with >95% validation accuracy
- Real-time detection using OpenCV
- REST API for image-based predictions
- Well-documented code and repository

## 9. References
- Kaggle Dataset: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
- TensorFlow Documentation: https://www.tensorflow.org/
- OpenCV Documentation: https://opencv.org/