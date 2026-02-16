"""
CNN Image Classification - Prediction Script
This script loads a trained CNN model and makes predictions on images in a specified folder.
It supports both JPG and PNG image formats.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

# Image dimensions (must match training configuration)
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Model path
MODEL_PATH = 'model.h5'
CLASS_INDICES_PATH = 'class_indices.json'

# Supported image formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

# ============================================================================
# LOAD MODEL AND CLASS INDICES
# ============================================================================

print("=" * 70)
print("CNN IMAGE CLASSIFICATION - PREDICTION")
print("=" * 70)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"\nError: Model file '{MODEL_PATH}' not found!")
    print("Please train the model first using 'train_model.py'")
    exit(1)

# Load the trained model
print(f"\nLoading model from '{MODEL_PATH}'...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Load class indices
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, 'r') as f:
        index_to_class = json.load(f)
    # Convert string keys back to integers
    index_to_class = {int(k): v for k, v in index_to_class.items()}
    print(f"Class indices loaded from '{CLASS_INDICES_PATH}'")
    print(f"Classes: {list(index_to_class.values())}")
else:
    print(f"\nWarning: '{CLASS_INDICES_PATH}' not found!")
    print("Predictions will show class indices instead of class names.")
    index_to_class = None

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def preprocess_image(img_path):
    """
    Load and preprocess an image for prediction.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Preprocessed image array ready for prediction
    """
    try:
        # Load image using PIL
        img = Image.open(img_path)
        
        # Convert to RGB if image is grayscale or has alpha channel
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image to target size
        img = img.resize(IMG_SIZE)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize pixel values to [0, 1] (same as training)
        img_array = img_array / 255.0
        
        # Add batch dimension (model expects batch of images)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        print(f"Error processing image '{img_path}': {str(e)}")
        return None


def predict_image(img_path):
    """
    Predict the class of a single image.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Tuple of (predicted_class, confidence)
    """
    # Preprocess the image
    img_array = preprocess_image(img_path)
    
    if img_array is None:
        return None, None
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    
    # Get the predicted class index
    predicted_index = np.argmax(predictions[0])
    
    # Get confidence score
    confidence = predictions[0][predicted_index]
    
    # Get class name if available
    if index_to_class is not None:
        predicted_class = index_to_class[predicted_index]
    else:
        predicted_class = f"Class_{predicted_index}"
    
    return predicted_class, confidence


def predict_folder(folder_path):
    """
    Predict classes for all images in a folder.
    
    Args:
        folder_path: Path to the folder containing images
    """
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"\nError: Folder '{folder_path}' not found!")
        return
    
    # Get all image files in the folder
    image_files = []
    for file in os.listdir(folder_path):
        file_ext = os.path.splitext(file)[1]
        if file_ext in SUPPORTED_FORMATS:
            image_files.append(file)
    
    # Check if any images were found
    if len(image_files) == 0:
        print(f"\nNo images found in '{folder_path}'")
        print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return
    
    print(f"\nFound {len(image_files)} images in '{folder_path}'")
    print("\n" + "=" * 70)
    print("PREDICTIONS")
    print("=" * 70)
    
    # Make predictions for each image
    successful_predictions = 0
    failed_predictions = 0
    
    for img_file in sorted(image_files):
        img_path = os.path.join(folder_path, img_file)
        
        # Predict
        predicted_class, confidence = predict_image(img_path)
        
        if predicted_class is not None:
            print(f"\nImage: {img_file}")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
            successful_predictions += 1
        else:
            print(f"\nImage: {img_file}")
            print("Prediction: FAILED")
            failed_predictions += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print(f"Total images: {len(image_files)}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Failed predictions: {failed_predictions}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Get folder path from user
    print("\n" + "=" * 70)
    print("ENTER FOLDER PATH FOR PREDICTION")
    print("=" * 70)
    print("\nSupported formats: JPG, JPEG, PNG")
    print("Example: test  or  predict_folder  or  C:/path/to/images")
    
    # Get input from user
    predict_folder_path = input("\nEnter folder path: ").strip()
    
    # Remove quotes if user added them
    predict_folder_path = predict_folder_path.strip('"').strip("'")
    
    # Make predictions
    predict_folder(predict_folder_path)
    
    print("\n" + "=" * 70)
    print("PREDICTION COMPLETE!")
    print("=" * 70)
