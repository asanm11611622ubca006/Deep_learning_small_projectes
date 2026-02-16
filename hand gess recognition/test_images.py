"""
==============================================================================
Test Hand Gesture Recognition Model on Sample Images
==============================================================================
This script loads the trained CNN model and tests it on images from a folder.
==============================================================================
"""

# Import required libraries
from tensorflow import keras
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
# Path to saved model
MODEL_PATH = r'd:\hand gess recognition\hand_gesture_model.h5'

# Path to folder containing test images
TEST_IMAGES_FOLDER = r'd:\hand gess recognition\sample.img'

# Image size (must match training size)
IMG_SIZE = (224, 224)

# Class labels (must match training order alphabetically)
CLASS_LABELS = ['FIVE', 'FOUR', 'NONE', 'ONE', 'THREE', 'TWO']

# Supported image extensions
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

# =============================================================================
# MAIN SCRIPT
# =============================================================================
print("=" * 70)
print("HAND GESTURE RECOGNITION - IMAGE TESTING")
print("=" * 70)

# Load the trained model
print(f"\n📦 Loading model from: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!\n")

# Get list of images from the folder
print(f"📂 Scanning folder: {TEST_IMAGES_FOLDER}")
image_files = [f for f in os.listdir(TEST_IMAGES_FOLDER) 
               if f.lower().endswith(SUPPORTED_EXTENSIONS)]

if not image_files:
    print("❌ No images found in the folder!")
    print(f"   Supported formats: {SUPPORTED_EXTENSIONS}")
else:
    print(f"✅ Found {len(image_files)} image(s)\n")
    
    # Print header
    print("=" * 70)
    print(f"{'IMAGE NAME':<25} {'PREDICTED CLASS':<15} {'CONFIDENCE':<12} {'RESULT'}")
    print("=" * 70)
    
    # Track correct predictions
    correct = 0
    total = 0
    
    # Process each image
    for image_name in sorted(image_files):
        image_path = os.path.join(TEST_IMAGES_FOLDER, image_name)
        
        try:
            # Load and preprocess the image
            img = keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize to [0, 1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            # Make prediction
            predictions = model.predict(img_array, verbose=0)
            predicted_index = np.argmax(predictions[0])
            predicted_class = CLASS_LABELS[predicted_index]
            confidence = predictions[0][predicted_index] * 100
            
            # Check if prediction matches actual label (from filename)
            actual_class = image_name.split('_')[0].upper()
            is_correct = predicted_class == actual_class
            result_emoji = "✅" if is_correct else "❌"
            
            if is_correct:
                correct += 1
            total += 1
            
            # Print result
            print(f"{image_name:<25} {predicted_class:<15} {confidence:>6.2f}%      {result_emoji}")
            
        except Exception as e:
            print(f"{image_name:<25} ERROR: {str(e)}")
    
    # Print summary
    print("=" * 70)
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n📊 SUMMARY:")
    print(f"   Total Images:  {total}")
    print(f"   Correct:       {correct}")
    print(f"   Accuracy:      {accuracy:.2f}%")
    print("=" * 70)

print("\n✅ Testing complete!")
