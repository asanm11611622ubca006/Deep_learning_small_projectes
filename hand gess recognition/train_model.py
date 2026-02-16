"""
==============================================================================
Hand Gesture Recognition using CNN with TensorFlow/Keras
==============================================================================

This script trains a Convolutional Neural Network (CNN) to classify hand gestures.
The model uses:
- CNN architecture with multiple convolutional layers
- Sigmoid activation in the output layer (multi-label compatible)
- ImageDataGenerator for loading and augmenting images
- Images resized to 224x224

Classes: FIVE, FOUR, NONE, ONE, THREE, TWO

Author: Hand Gesture Recognition Project
==============================================================================
"""

# ==============================================================================
# REQUIRED PYTHON LIBRARIES
# ==============================================================================
# Install these libraries before running the script:
# pip install tensorflow
# pip install numpy
# pip install matplotlib
# pip install pillow
#
# For GPU support (optional):
# pip install tensorflow-gpu
# ==============================================================================

# =============================================================================
# IMPORT REQUIRED LIBRARIES
# =============================================================================

import os  # For file and directory operations
import numpy as np  # For numerical operations
import tensorflow as tf  # Deep learning framework
from tensorflow import keras  # High-level neural network API
from tensorflow.keras.models import Sequential  # For creating sequential models
from tensorflow.keras.layers import (
    Conv2D,          # 2D Convolutional layer
    MaxPooling2D,    # Max pooling layer for downsampling
    Flatten,         # Flatten layer to convert 2D to 1D
    Dense,           # Fully connected layer
    Dropout,         # Dropout layer for regularization
    BatchNormalization  # Batch normalization for faster training
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For loading images
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Training callbacks
import matplotlib.pyplot as plt  # For plotting graphs

# =============================================================================
# CONFIGURATION AND HYPERPARAMETERS
# =============================================================================

# Image dimensions - images will be resized to this size
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Training parameters
BATCH_SIZE = 32  # Number of images processed in each training step
EPOCHS = 5  # Number of complete passes through the training data (reduced for faster training)

# Number of classes in the dataset
NUM_CLASSES = 6  # FIVE, FOUR, NONE, ONE, THREE, TWO

# Dataset paths
TRAIN_DIR = r"d:\hand gess recognition\train"  # Path to training data
TEST_DIR = r"d:\hand gess recognition\test"    # Path to test/validation data

# Model save path
MODEL_SAVE_PATH = r"d:\hand gess recognition\hand_gesture_model.h5"

# =============================================================================
# CREATE IMAGE DATA GENERATORS
# =============================================================================

print("=" * 60)
print("HAND GESTURE RECOGNITION - CNN MODEL TRAINING")
print("=" * 60)
print(f"\nTensorFlow Version: {tf.__version__}")
print(f"\nConfiguration:")
print(f"  - Image Size: {IMG_WIDTH}x{IMG_HEIGHT}")
print(f"  - Batch Size: {BATCH_SIZE}")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Number of Classes: {NUM_CLASSES}")
print()

# -------------------------------------------------------------------------
# Training Data Generator with Data Augmentation
# -------------------------------------------------------------------------
# Data augmentation helps prevent overfitting by creating variations
# of the training images (rotation, zoom, shift, etc.)

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,           # Normalize pixel values to [0, 1]
    rotation_range=20,            # Random rotation up to 20 degrees
    width_shift_range=0.2,        # Random horizontal shift
    height_shift_range=0.2,       # Random vertical shift
    shear_range=0.15,             # Random shear transformation
    zoom_range=0.15,              # Random zoom
    horizontal_flip=True,         # Random horizontal flip
    fill_mode='nearest'           # Fill mode for newly created pixels
)

# -------------------------------------------------------------------------
# Test/Validation Data Generator (No Augmentation)
# -------------------------------------------------------------------------
# Test data should not be augmented - only rescale

test_datagen = ImageDataGenerator(
    rescale=1.0/255.0  # Only normalize pixel values
)

# -------------------------------------------------------------------------
# Load Training Images using flow_from_directory
# -------------------------------------------------------------------------
print("Loading training data...")

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,                    # Directory containing training images
    target_size=IMG_SIZE,         # Resize images to 224x224
    batch_size=BATCH_SIZE,        # Number of images per batch
    class_mode='categorical',     # Multi-class classification
    shuffle=True,                 # Shuffle training data
    color_mode='rgb'              # Load as RGB images
)

# -------------------------------------------------------------------------
# Load Test/Validation Images
# -------------------------------------------------------------------------
print("Loading test/validation data...")

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,                     # Directory containing test images
    target_size=IMG_SIZE,         # Resize images to 224x224
    batch_size=BATCH_SIZE,        # Number of images per batch
    class_mode='categorical',     # Multi-class classification
    shuffle=False,                # Don't shuffle test data
    color_mode='rgb'              # Load as RGB images
)

# Print class labels
print("\nClass Labels:")
class_labels = train_generator.class_indices
for class_name, index in class_labels.items():
    print(f"  {index}: {class_name}")

# =============================================================================
# BUILD CNN MODEL ARCHITECTURE
# =============================================================================

print("\n" + "=" * 60)
print("BUILDING CNN MODEL ARCHITECTURE")
print("=" * 60)

def build_cnn_model():
    """
    Build a CNN model for hand gesture recognition.
    
    Architecture:
    - 4 Convolutional blocks (Conv2D + BatchNorm + MaxPool)
    - Flatten layer
    - 2 Dense layers with Dropout
    - Output layer with Sigmoid activation
    
    Returns:
        keras.Model: Compiled CNN model
    """
    
    model = Sequential([
        # ------------------------------------------------------------------
        # FIRST CONVOLUTIONAL BLOCK
        # Input: 224x224x3 (RGB image)
        # ------------------------------------------------------------------
        # Conv2D: 32 filters, 3x3 kernel, ReLU activation
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        BatchNormalization(),  # Normalize activations for faster training
        MaxPooling2D(pool_size=(2, 2)),  # Downsample by factor of 2
        # Output: 112x112x32
        
        # ------------------------------------------------------------------
        # SECOND CONVOLUTIONAL BLOCK
        # ------------------------------------------------------------------
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        # Output: 56x56x64
        
        # ------------------------------------------------------------------
        # THIRD CONVOLUTIONAL BLOCK
        # ------------------------------------------------------------------
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        # Output: 28x28x128
        
        # ------------------------------------------------------------------
        # FOURTH CONVOLUTIONAL BLOCK
        # ------------------------------------------------------------------
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        # Output: 14x14x256
        
        # ------------------------------------------------------------------
        # FIFTH CONVOLUTIONAL BLOCK
        # ------------------------------------------------------------------
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        # Output: 7x7x512
        
        # ------------------------------------------------------------------
        # FLATTEN AND DENSE LAYERS
        # ------------------------------------------------------------------
        Flatten(),  # Convert 2D feature maps to 1D vector
        # Output: 7*7*512 = 25088 neurons
        
        # First Dense Layer
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),  # 50% dropout to prevent overfitting
        
        # Second Dense Layer
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # ------------------------------------------------------------------
        # OUTPUT LAYER
        # Using Sigmoid activation as requested
        # ------------------------------------------------------------------
        Dense(NUM_CLASSES, activation='sigmoid')
        # For multi-class with sigmoid, each output represents probability
        # of belonging to that class
    ])
    
    return model

# Create the model
model = build_cnn_model()

# =============================================================================
# COMPILE THE MODEL
# =============================================================================

# Note: Since we're using sigmoid with multi-class classification,
# we use categorical_crossentropy loss which works with sigmoid
# by treating it as independent binary classifications
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Adam optimizer
    loss='categorical_crossentropy',  # Loss function for multi-class
    metrics=['accuracy']  # Track accuracy during training
)

# =============================================================================
# DISPLAY MODEL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)
model.summary()

# =============================================================================
# DEFINE CALLBACKS
# =============================================================================

# Custom Callback: Stop training when accuracy reaches 90%
class AccuracyThresholdCallback(keras.callbacks.Callback):
    """Custom callback to stop training when accuracy reaches threshold."""
    def __init__(self, threshold=0.90):
        super().__init__()
        self.threshold = threshold
    
    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy')
        if accuracy is not None and accuracy >= self.threshold:
            print(f"\n\nTraining accuracy reached {accuracy*100:.2f}% (>= {self.threshold*100}%)")
            print("Stopping training and saving model...")
            self.model.stop_training = True

# Accuracy threshold callback: Stop at 90% training accuracy
accuracy_callback = AccuracyThresholdCallback(threshold=0.90)

# Early Stopping: Stop training if validation loss doesn't improve
early_stopping = EarlyStopping(
    monitor='val_loss',           # Monitor validation loss
    patience=3,                   # Wait 3 epochs before stopping
    restore_best_weights=True,    # Restore best model weights
    verbose=1
)

# Learning Rate Reduction: Reduce LR when validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',           # Monitor validation loss
    factor=0.2,                   # Reduce LR by factor of 0.2
    patience=2,                   # Wait 2 epochs before reducing
    min_lr=1e-6,                  # Minimum learning rate
    verbose=1
)

callbacks = [accuracy_callback, early_stopping, reduce_lr]

# =============================================================================
# TRAIN THE MODEL
# =============================================================================

print("\n" + "=" * 60)
print("TRAINING THE MODEL")
print("=" * 60)

# Calculate steps per epoch
steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = test_generator.samples // BATCH_SIZE

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {test_generator.samples}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")
print("\nStarting training...\n")

# Train the model
history = model.fit(
    train_generator,                      # Training data generator
    epochs=EPOCHS,                        # Number of epochs
    steps_per_epoch=steps_per_epoch,      # Steps per epoch
    validation_data=test_generator,       # Validation data
    validation_steps=validation_steps,    # Validation steps
    callbacks=callbacks,                  # Training callbacks
    verbose=1                             # Show progress
)

# =============================================================================
# SAVE THE TRAINED MODEL
# =============================================================================

print("\n" + "=" * 60)
print("SAVING THE MODEL")
print("=" * 60)

model.save(MODEL_SAVE_PATH)
print(f"\nModel saved successfully to: {MODEL_SAVE_PATH}")

# =============================================================================
# EVALUATE THE MODEL ON TEST DATA
# =============================================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Reset the test generator
test_generator.reset()

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(
    test_generator,
    steps=validation_steps,
    verbose=1
)

# =============================================================================
# DISPLAY FINAL RESULTS
# =============================================================================

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

# Get final training metrics from history
final_train_accuracy = history.history['accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"\n{'Metric':<25} {'Training':<15} {'Validation':<15}")
print("-" * 55)
print(f"{'Accuracy':<25} {final_train_accuracy:.4f} ({final_train_accuracy*100:.2f}%)   {final_val_accuracy:.4f} ({final_val_accuracy*100:.2f}%)")
print(f"{'Loss':<25} {final_train_loss:.4f}            {final_val_loss:.4f}")
print()
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Loss: {test_loss:.4f}")

# Best epoch based on validation accuracy
best_val_acc_epoch = np.argmax(history.history['val_accuracy']) + 1
best_val_acc = max(history.history['val_accuracy'])
print(f"\nBest Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%) at Epoch {best_val_acc_epoch}")

# =============================================================================
# PLOT TRAINING HISTORY
# =============================================================================

print("\n" + "=" * 60)
print("GENERATING TRAINING PLOTS")
print("=" * 60)

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# -------------------------------------------------------------------------
# Plot 1: Training & Validation Accuracy
# -------------------------------------------------------------------------
axes[0].plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
axes[0].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1])

# -------------------------------------------------------------------------
# Plot 2: Training & Validation Loss
# -------------------------------------------------------------------------
axes[1].plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
axes[1].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plot_path = r"d:\hand gess recognition\training_history.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nTraining plots saved to: {plot_path}")

plt.show()

# =============================================================================
# DISPLAY CLASS-WISE INFORMATION
# =============================================================================

print("\n" + "=" * 60)
print("CLASS MAPPING")
print("=" * 60)
print("\nTo use this model for prediction, use the following class mapping:")
for class_name, index in sorted(class_labels.items(), key=lambda x: x[1]):
    print(f"  Class {index}: {class_name}")

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\nModel saved as: {MODEL_SAVE_PATH}")
print(f"You can load this model using:")
print(f"  model = keras.models.load_model('{MODEL_SAVE_PATH}')")
print()
