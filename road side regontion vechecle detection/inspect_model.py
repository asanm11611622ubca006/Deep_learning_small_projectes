"""
Model Inspector Script
This script inspects the trained model to understand its architecture,
input shape, and output classes.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
import io

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def inspect_model(model_path):
    """
    Inspect a trained Keras model and print its details.
    """
    print("=" * 60)
    print("MODEL INSPECTION REPORT")
    print("=" * 60)
    
    # Load the model
    try:
        model = keras.models.load_model(model_path)
        print("\n[OK] Model loaded successfully!")
    except Exception as e:
        print(f"\n[ERROR] Error loading model: {e}")
        return None
    
    # Model Summary
    print("\n" + "-" * 40)
    print("MODEL ARCHITECTURE SUMMARY")
    print("-" * 40)
    model.summary()
    
    # Input Shape
    print("\n" + "-" * 40)
    print("INPUT DETAILS")
    print("-" * 40)
    try:
        input_shape = model.input_shape
    except:
        try:
            input_shape = model.layers[0].input_shape
        except:
            input_shape = None
    
    print(f"Input Shape: {input_shape}")
    if input_shape and len(input_shape) == 4:
        print(f"  - Batch Size: {input_shape[0]} (None = variable)")
        print(f"  - Image Height: {input_shape[1]} pixels")
        print(f"  - Image Width: {input_shape[2]} pixels")
        print(f"  - Channels: {input_shape[3]} (3 = RGB, 1 = Grayscale)")
    
    # Output Shape
    print("\n" + "-" * 40)
    print("OUTPUT DETAILS")
    print("-" * 40)
    try:
        output_shape = model.output_shape
    except:
        try:
            output_shape = model.layers[-1].output.shape
        except:
            output_shape = None
    
    print(f"Output Shape: {output_shape}")
    if output_shape and len(output_shape) == 2:
        num_classes = output_shape[1]
        print(f"  - Number of Classes: {num_classes}")
    
    # Layer Information
    print("\n" + "-" * 40)
    print("LAYER BREAKDOWN")
    print("-" * 40)
    for i, layer in enumerate(model.layers):
        layer_name = layer.name
        layer_type = layer.__class__.__name__
        try:
            out_shape = layer.output.shape
        except:
            out_shape = "N/A"
        print(f"{i+1}. {layer_name}: {layer_type} | Output: {out_shape}")
    
    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)
    
    return model

if __name__ == "__main__":
    model_path = r"d:\road side regontion vechecle detection\my_model.h5"
    model = inspect_model(model_path)
