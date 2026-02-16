import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

# Set parameters
IMG_HEIGHT = 28
IMG_WIDTH = 28
BATCH_SIZE = 32
EPOCHS = 22
TRAIN_DIR = 'train'

def train_character_recognition():
    # check if train dir exists
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Directory '{TRAIN_DIR}' not found.")
        return

    # 1. Load dataset using ImageDataGenerator
    # Rescale pixels to 0-1
    # We use validation_split to create a validation set from the training data
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    print("Loading training data...")
    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        subset='training'
    )

    print("Loading validation data...")
    validation_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        subset='validation'
    )

    # Save class indices for the GUI app
    class_indices = train_generator.class_indices
    # Invert to map index -> label
    label_map = {v: k for k, v in class_indices.items()}
    with open('class_indices.json', 'w') as f:
        json.dump(label_map, f)
    print("Class indices saved to class_indices.json")

    # 2. Build CNN model
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        MaxPooling2D(2, 2),

        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # Flatten and Dense Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Add dropout to prevent overfitting
        Dense(len(class_indices), activation='softmax') # Output layer size = number of classes
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # 3. Train the model
    print("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS
    )

    # 4. Save the model
    model.save('character_cnn.h5')
    print("Model saved as character_cnn.h5")

    # 5. Print final accuracy
    final_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    print(f"\nTraining completed.")
    print(f"Final Training Accuracy: {final_acc*100:.2f}%")
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")

if __name__ == "__main__":
    train_character_recognition()
