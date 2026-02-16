"""
==============================================================================
Load and Test Hand Gesture Recognition Model
==============================================================================
This script loads the trained CNN model and displays its summary.
==============================================================================
"""

# Import required libraries
from tensorflow import keras
import numpy as np

# Path to saved model
MODEL_PATH = r'd:\hand gess recognition\hand_gesture_model.h5'

# Class labels (must match training order)
CLASS_LABELS = ['FIVE', 'FOUR', 'NONE', 'ONE', 'THREE', 'TWO']

print("=" * 60)
print("LOADING HAND GESTURE RECOGNITION MODEL")
print("=" * 60)

# Load the model
print(f"\nLoading model from: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)

print("\n✅ Model loaded successfully!")

# Display model summary
print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)
model.summary()

# Display class labels
print("\n" + "=" * 60)
print("CLASS LABELS")
print("=" * 60)
for i, label in enumerate(CLASS_LABELS):
    print(f"  Class {i}: {label}")

print("\n" + "=" * 60)
print("MODEL READY FOR PREDICTIONS!")
print("=" * 60)
print("\nTo make a prediction on an image, use:")
print("  img = keras.preprocessing.image.load_img('path/to/image.png', target_size=(224, 224))")
print("  img_array = keras.preprocessing.image.img_to_array(img) / 255.0")
print("  img_array = np.expand_dims(img_array, axis=0)")
print("  predictions = model.predict(img_array)")
print("  predicted_class = CLASS_LABELS[np.argmax(predictions)]")
