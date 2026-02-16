# CNN Image Classification Project

A beginner-friendly Python project for image classification using Convolutional Neural Networks (CNN) with TensorFlow and Keras.

## 📋 Project Overview

This project trains a CNN model to classify images into different categories. It includes:

- **Training script** (`train_model.py`): Trains the CNN model using your dataset
- **Prediction script** (`predict.py`): Makes predictions on new images
- **Automatic model saving**: Saves the best model during training
- **Data augmentation**: Improves model generalization
- **Visualization**: Generates training history plots

## 📁 Project Structure

```
Image classification using_CNN/
├── train/                  # Training images (organized by class subfolders)
├── val/                    # Validation images (organized by class subfolders)
├── test/                   # Test images (organized by class subfolders)
├── train_model.py          # Training script
├── predict.py              # Prediction script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── model.h5               # Trained model (generated after training)
├── class_indices.json     # Class mappings (generated after training)
└── training_history.png   # Training plots (generated after training)
```

## 🔧 Installation

### Step 1: Install Python

Make sure you have Python 3.8 or higher installed on your system.

### Step 2: Install Dependencies

Open a terminal/command prompt in the project directory and run:

```bash
pip install -r requirements.txt
```

This will install:

- TensorFlow (deep learning framework)
- NumPy (numerical operations)
- Matplotlib (plotting)
- Pillow (image processing)

## 📊 Dataset Structure

Your dataset folders should be organized as follows:

```
train/
├── class1/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── class3/
    └── ...

val/
├── class1/
├── class2/
└── class3/

test/
├── class1/
├── class2/
└── class3/
```

**Important Notes:**

- Each subfolder name represents a class name
- Supported formats: JPG, JPEG, PNG
- Images will be automatically resized to 224x224 pixels
- Images will be automatically normalized to [0, 1] range

## 🚀 Usage

### Training the Model

Run the training script:

```bash
python train_model.py
```

**What happens during training:**

1. Loads images from `train/` and `val/` folders
2. Applies data augmentation to training images
3. Builds a CNN model with multiple convolutional layers
4. Trains the model for up to 50 epochs (with early stopping)
5. Saves the best model to `model.h5`
6. Saves class mappings to `class_indices.json`
7. Generates training history plot to `training_history.png`

**Training Parameters:**

- Image size: 224x224 pixels
- Batch size: 32
- Maximum epochs: 50
- Learning rate: 0.001
- Early stopping: Stops if no improvement for 10 epochs

**Expected Output:**

```
======================================================================
CNN IMAGE CLASSIFICATION - TRAINING
======================================================================

Loading training data from 'train'...
Found 1000 images belonging to 5 classes.

Loading validation data from 'val'...
Found 200 images belonging to 5 classes.

Number of classes detected: 5
Class names: ['class1', 'class2', 'class3', 'class4', 'class5']
...
```

### Making Predictions

After training, run the prediction script:

```bash
python predict.py
```

**How to use:**

1. The script will prompt you to enter a folder path
2. Enter the path to a folder containing images (e.g., `test` or `predict_folder`)
3. The script will process all JPG/PNG images in that folder
4. For each image, it will display:
   - Image filename
   - Predicted class name
   - Confidence score (percentage)

**Example Output:**

```
======================================================================
CNN IMAGE CLASSIFICATION - PREDICTION
======================================================================

Loading model from 'model.h5'...
Model loaded successfully!

Found 10 images in 'test'

======================================================================
PREDICTIONS
======================================================================

Image: cat_001.jpg
Predicted Class: cat
Confidence: 95.67%

Image: dog_001.jpg
Predicted Class: dog
Confidence: 89.23%
...
```

## 🏗️ Model Architecture

The CNN model consists of:

1. **4 Convolutional Blocks:**
   - Conv2D layers with ReLU activation
   - Batch Normalization (improves training stability)
   - MaxPooling (reduces spatial dimensions)
   - Dropout (prevents overfitting)

2. **Dense Layers:**
   - Flatten layer (converts 2D features to 1D)
   - 2 Dense layers with dropout
   - Output layer with softmax activation

**Key Features:**

- **Data Augmentation**: Rotation, shifting, flipping, zooming
- **Batch Normalization**: Faster and more stable training
- **Dropout**: Reduces overfitting
- **Early Stopping**: Prevents overtraining
- **Model Checkpointing**: Saves the best model automatically

## 📈 Understanding the Results

### Training History Plot

After training, check `training_history.png` to see:

- **Accuracy plot**: Shows training and validation accuracy over epochs
- **Loss plot**: Shows training and validation loss over epochs

**Good signs:**

- Validation accuracy increases over time
- Gap between training and validation accuracy is small
- Loss decreases steadily

**Warning signs:**

- Validation accuracy plateaus or decreases (overfitting)
- Large gap between training and validation accuracy (overfitting)

### Prediction Confidence

- **>90%**: Very confident prediction
- **70-90%**: Confident prediction
- **50-70%**: Moderate confidence
- **<50%**: Low confidence (may be incorrect)

## 🛠️ Troubleshooting

### Issue: "No module named 'tensorflow'"

**Solution:** Install TensorFlow: `pip install tensorflow`

### Issue: "Found 0 images"

**Solution:**

- Check that your dataset folders exist
- Ensure images are in subfolders (not directly in train/val/test)
- Verify image formats are JPG or PNG

### Issue: Model accuracy is low

**Solutions:**

- Collect more training data
- Ensure dataset is balanced (similar number of images per class)
- Train for more epochs
- Adjust learning rate or model architecture

### Issue: "Out of memory" error

**Solutions:**

- Reduce batch size in `train_model.py` (e.g., change to 16 or 8)
- Close other applications
- Use a machine with more RAM/GPU

## 💡 Tips for Better Results

1. **More Data**: Collect at least 100-200 images per class
2. **Balanced Dataset**: Each class should have similar number of images
3. **Quality Images**: Use clear, well-lit images
4. **Diverse Images**: Include variations in angle, lighting, background
5. **Clean Data**: Remove corrupted or mislabeled images

## 📝 Customization

### Change Image Size

Edit both `train_model.py` and `predict.py`:

```python
IMG_HEIGHT = 224  # Change to your desired size
IMG_WIDTH = 224
```

### Adjust Training Parameters

Edit `train_model.py`:

```python
BATCH_SIZE = 32      # Reduce if out of memory
EPOCHS = 50          # Increase for more training
LEARNING_RATE = 0.001  # Adjust learning speed
```

### Modify Model Architecture

Edit the model definition in `train_model.py` to add/remove layers.

## 📚 Learning Resources

- [TensorFlow Documentation](https://www.tensorflow.org/tutorials)
- [Keras Documentation](https://keras.io/guides/)
- [CNN Explained](https://cs231n.github.io/convolutional-networks/)

## 🤝 Support

If you encounter any issues:

1. Check the error message carefully
2. Verify your dataset structure
3. Ensure all dependencies are installed
4. Check that Python version is 3.8+

## 📄 License

This project is open-source and free to use for educational purposes.

---

**Happy Learning! 🎉**
