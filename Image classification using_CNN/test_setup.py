"""
Quick test to verify TensorFlow and dataset are working correctly
"""

import os
import sys

print("=" * 70)
print("TESTING CNN PROJECT SETUP")
print("=" * 70)

# Test 1: Import libraries
print("\n[1/5] Testing library imports...")
try:
    import tensorflow as tf
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from PIL import Image
    print("✓ All libraries imported successfully!")
    print(f"  - TensorFlow version: {tf.__version__}")
    print(f"  - NumPy version: {np.__version__}")
except Exception as e:
    print(f"✗ Error importing libraries: {e}")
    sys.exit(1)

# Test 2: Check dataset folders
print("\n[2/5] Checking dataset folders...")
folders = ['train', 'val', 'test']
for folder in folders:
    if os.path.exists(folder):
        print(f"✓ Found '{folder}' folder")
    else:
        print(f"✗ Missing '{folder}' folder")

# Test 3: Check classes
print("\n[3/5] Checking classes in train folder...")
if os.path.exists('train'):
    classes = [d for d in os.listdir('train') if os.path.isdir(os.path.join('train', d))]
    print(f"✓ Found {len(classes)} classes: {classes}")
    
    for cls in classes:
        cls_path = os.path.join('train', cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  - {cls}: {len(images)} images")
else:
    print("✗ Train folder not found")

# Test 4: Test ImageDataGenerator
print("\n[4/5] Testing ImageDataGenerator...")
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        'train',
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical'
    )
    print(f"✓ ImageDataGenerator working!")
    print(f"  - Found {generator.samples} images")
    print(f"  - Classes: {list(generator.class_indices.keys())}")
except Exception as e:
    print(f"✗ Error with ImageDataGenerator: {e}")
    sys.exit(1)

# Test 5: Test loading a single image
print("\n[5/5] Testing image loading...")
try:
    # Find first image
    for cls in os.listdir('train'):
        cls_path = os.path.join('train', cls)
        if os.path.isdir(cls_path):
            images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                test_img_path = os.path.join(cls_path, images[0])
                img = Image.open(test_img_path)
                img_array = np.array(img)
                print(f"✓ Successfully loaded test image: {images[0]}")
                print(f"  - Original size: {img.size}")
                print(f"  - Array shape: {img_array.shape}")
                break
except Exception as e:
    print(f"✗ Error loading image: {e}")

print("\n" + "=" * 70)
print("SETUP TEST COMPLETE!")
print("=" * 70)
print("\nIf all tests passed, you can run: python train_model.py")
