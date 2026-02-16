"""
QUICK START EXAMPLE - Prediction Script
This is a simplified version that you can use to quickly test predictions
without entering folder paths manually.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# Configuration
IMG_SIZE = (224, 224)
MODEL_PATH = 'model.h5'
CLASS_INDICES_PATH = 'class_indices.json'
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

def predict_folder_quick(folder_path):
    """Quick prediction function for a folder of images."""
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model '{MODEL_PATH}' not found! Train the model first.")
        return
    
    model = keras.models.load_model(MODEL_PATH)
    
    # Load class indices
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, 'r') as f:
            index_to_class = {int(k): v for k, v in json.load(f).items()}
    else:
        index_to_class = None
    
    # Get image files
    image_files = [f for f in os.listdir(folder_path) 
                   if os.path.splitext(f)[1] in SUPPORTED_FORMATS]
    
    if not image_files:
        print(f"No images found in '{folder_path}'")
        return
    
    print(f"\n{'='*70}")
    print(f"PREDICTIONS FOR: {folder_path}")
    print(f"{'='*70}\n")
    
    # Predict each image
    for img_file in sorted(image_files):
        img_path = os.path.join(folder_path, img_file)
        
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array, verbose=0)
            predicted_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_index]
            
            # Get class name
            if index_to_class:
                predicted_class = index_to_class[predicted_index]
            else:
                predicted_class = f"Class_{predicted_index}"
            
            print(f"{img_file:30s} → {predicted_class:20s} ({confidence:.1%})")
            
        except Exception as e:
            print(f"{img_file:30s} → ERROR: {str(e)}")
    
    print(f"\n{'='*70}\n")

# Example usage - modify the folder path below
if __name__ == "__main__":
    # Option 1: Use test folder
    predict_folder_quick('test')
    
    # Option 2: Specify custom folder (uncomment to use)
    # predict_folder_quick('predict_folder')
    
    # Option 3: Specify full path (uncomment to use)
    # predict_folder_quick('C:/path/to/your/images')
