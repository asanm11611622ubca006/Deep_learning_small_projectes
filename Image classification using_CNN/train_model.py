"""
CNN Image Classification - Training Script
This script trains a Convolutional Neural Network (CNN) model for image classification
using TensorFlow and Keras with ImageDataGenerator for data augmentation.
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

# Image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Dataset paths
TRAIN_DIR = 'train'
VAL_DIR = 'val'
TEST_DIR = 'test'

# Model save path
MODEL_PATH = 'model.h5'

# ============================================================================
# DATA PREPARATION
# ============================================================================

print("=" * 70)
print("CNN IMAGE CLASSIFICATION - TRAINING")
print("=" * 70)

# Create ImageDataGenerator for training data with normalization and augmentation
# Normalization: rescale pixel values from [0, 255] to [0, 1]
# Data augmentation helps prevent overfitting by creating variations of training images
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values to [0, 1]
    rotation_range=20,           # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,       # Randomly shift images horizontally
    height_shift_range=0.2,      # Randomly shift images vertically
    horizontal_flip=True,        # Randomly flip images horizontally
    zoom_range=0.2,              # Randomly zoom into images
    shear_range=0.2,             # Randomly apply shearing transformations
    fill_mode='nearest'          # Fill in newly created pixels
)

# Create ImageDataGenerator for validation data (only normalization, no augmentation)
val_datagen = ImageDataGenerator(
    rescale=1./255               # Only normalize, no augmentation for validation
)

# Load training data from directory
# Each subfolder in TRAIN_DIR represents a class
print(f"\nLoading training data from '{TRAIN_DIR}'...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,        # Resize all images to 224x224
    batch_size=BATCH_SIZE,
    class_mode='categorical',    # Multi-class classification
    shuffle=True                 # Shuffle training data
)

# Load validation data from directory
print(f"Loading validation data from '{VAL_DIR}'...")
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,        # Resize all images to 224x224
    batch_size=BATCH_SIZE,
    class_mode='categorical',    # Multi-class classification
    shuffle=False                # Don't shuffle validation data
)

# Get number of classes
num_classes = len(train_generator.class_indices)
print(f"\nNumber of classes detected: {num_classes}")
print(f"Class names: {list(train_generator.class_indices.keys())}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

print("\n" + "=" * 70)
print("BUILDING CNN MODEL")
print("=" * 70)

# Build a Sequential CNN model
model = keras.Sequential([
    # Input layer
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Third Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Fourth Convolutional Block
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    # Output layer (softmax for multi-class classification)
    layers.Dense(num_classes, activation='softmax')
])

# Display model architecture
model.summary()

# ============================================================================
# MODEL COMPILATION
# ============================================================================

print("\n" + "=" * 70)
print("COMPILING MODEL")
print("=" * 70)

# Compile the model
# Adam optimizer: adaptive learning rate optimization algorithm
# Categorical crossentropy: loss function for multi-class classification
# Accuracy: metric to monitor during training
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully!")

# ============================================================================
# CALLBACKS
# ============================================================================

# ModelCheckpoint: Save the best model during training
checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor='val_accuracy',      # Monitor validation accuracy
    save_best_only=True,         # Only save when validation accuracy improves
    mode='max',                  # Save model with maximum validation accuracy
    verbose=1
)

# EarlyStopping: Stop training if validation accuracy doesn't improve
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,                 # Stop if no improvement for 10 epochs
    restore_best_weights=True,   # Restore weights from best epoch
    verbose=1
)

# ============================================================================
# MODEL TRAINING
# ============================================================================

print("\n" + "=" * 70)
print("TRAINING MODEL")
print("=" * 70)

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# ============================================================================
# SAVE FINAL MODEL
# ============================================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

# Save the final model
model.save(MODEL_PATH)
print(f"Model saved successfully to '{MODEL_PATH}'")

# Save class indices for later use in prediction
import json
class_indices = train_generator.class_indices
# Reverse the dictionary to map index to class name
index_to_class = {v: k for k, v in class_indices.items()}

with open('class_indices.json', 'w') as f:
    json.dump(index_to_class, f, indent=4)
print("Class indices saved to 'class_indices.json'")

# ============================================================================
# PLOT TRAINING HISTORY
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING TRAINING PLOTS")
print("=" * 70)

# Plot training and validation accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("Training history plot saved to 'training_history.png'")

# ============================================================================
# FINAL EVALUATION
# ============================================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)

# Get final training and validation accuracy
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print(f"\nModel saved to: {MODEL_PATH}")
print("You can now use 'predict.py' to make predictions on new images!")
