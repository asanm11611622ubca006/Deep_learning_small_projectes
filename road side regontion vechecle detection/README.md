# 🚦 Traffic Sign Recognition System

A professional, AI-powered desktop application for recognizing traffic signs from images. This system uses a pre-trained Convolutional Neural Network (CNN) to classify **43 different traffic sign types** from the German Traffic Sign Recognition Benchmark.

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `traffic_sign_gui.py` | **Main GUI Application** - Professional dark-themed interface for image upload and prediction |
| `dataset_visualizer.py` | **Dataset Analysis Tool** - Generate beautiful charts and statistics for your training data |
| `my_model.h5` | **Pre-trained Model** - CNN model trained on traffic sign dataset |
| `inspect_model.py` | **Model Inspector** - Utility to view model architecture details |

---

## 🚀 How to Run

### Run the Main GUI Application
```bash
cd "d:\road side regontion vechecle detection"
python traffic_sign_gui.py
```

### Run Dataset Visualizer (for graphs)
```bash
python dataset_visualizer.py
```
Then enter the path to your training dataset when prompted.

---

## 🎯 Supported Traffic Sign Classes (43 Types)

| ID | Sign Name | ID | Sign Name |
|----|-----------|----|-----------| 
| 0 | Speed limit (20km/h) | 22 | Bumpy road |
| 1 | Speed limit (30km/h) | 23 | Slippery road |
| 2 | Speed limit (50km/h) | 24 | Road narrows on the right |
| 3 | Speed limit (60km/h) | 25 | Road work |
| 4 | Speed limit (70km/h) | 26 | Traffic signals |
| 5 | Speed limit (80km/h) | 27 | Pedestrians |
| 6 | End of speed limit (80km/h) | 28 | Children crossing |
| 7 | Speed limit (100km/h) | 29 | Bicycles crossing |
| 8 | Speed limit (120km/h) | 30 | Beware of ice/snow |
| 9 | No passing | 31 | Wild animals crossing |
| 10 | No passing (vehicles >3.5t) | 32 | End of all speed/passing limits |
| 11 | Right-of-way at intersection | 33 | Turn right ahead |
| 12 | Priority road | 34 | Turn left ahead |
| 13 | Yield | 35 | Ahead only |
| 14 | Stop | 36 | Go straight or right |
| 15 | No vehicles | 37 | Go straight or left |
| 16 | Vehicles >3.5t prohibited | 38 | Keep right |
| 17 | No entry | 39 | Keep left |
| 18 | General caution | 40 | Roundabout mandatory |
| 19 | Dangerous curve left | 41 | End of no passing |
| 20 | Dangerous curve right | 42 | End of no passing (>3.5t) |
| 21 | Double curve | | |

---

## 🖼️ GUI Features

### Main Interface
- **🎨 Modern Dark Theme** - Professional glassmorphism-inspired design
- **📤 Easy Image Upload** - Click to browse or drag-and-drop images
- **🔍 Instant Prediction** - Real-time traffic sign classification
- **📊 Confidence Score** - Visual progress bar showing prediction confidence
- **🏆 Top 5 Predictions** - See the 5 most likely sign classes

### How to Use
1. Launch the application using `python traffic_sign_gui.py`
2. Click "Select Image" or click the upload area
3. Choose a traffic sign image (JPG, PNG, BMP, GIF supported)
4. The prediction will appear automatically!

---

## 📊 Dataset Visualization

The `dataset_visualizer.py` script generates beautiful charts:

### Generated Charts
1. **Class Distribution Bar Chart** - Shows number of images per class
2. **Pie Chart** - Top 10 classes by sample count
3. **Sample Images Grid** - Preview of images from each class
4. **Statistics Summary** - Comprehensive overview dashboard

### Usage
```python
python dataset_visualizer.py
# Enter your dataset path when prompted
# Example: D:\datasets\traffic_signs\train
```

---

## 🧠 Model Architecture

```
Model: Sequential CNN
├── Conv2D (32 filters, 5x5)
├── Conv2D (32 filters, 5x5)
├── MaxPooling2D (2x2)
├── Dropout (0.25)
├── Conv2D (64 filters, 3x3)
├── Conv2D (64 filters, 3x3)
├── MaxPooling2D (2x2)
├── Dropout (0.25)
├── Flatten
├── Dense (256 neurons)
├── Dropout (0.5)
└── Dense (43 classes, softmax)

Input: 30x30x3 RGB images
Output: 43-class probability distribution
Total Parameters: 242,253
```

---

## 📦 Requirements

```
tensorflow>=2.0
numpy
Pillow
matplotlib
seaborn
```

Install all requirements:
```bash
pip install tensorflow numpy Pillow matplotlib seaborn
```

---

## 🔧 Troubleshooting

### "Model not loaded" error
Ensure `my_model.h5` is in the same directory as `traffic_sign_gui.py`

### Image not predicting correctly
- Use clear, well-lit images of traffic signs
- The model works best with cropped sign images (not full road scenes)
- Images are resized to 30x30, so very detailed signs may lose information

### GUI not appearing
- Make sure Tkinter is installed: `pip install tk`
- On some systems: `sudo apt-get install python3-tk`

---

## 📝 Simple Explanation: How It Works

### Step 1: Image Upload
When you upload an image, it gets read into memory using the **Pillow** library.

### Step 2: Preprocessing
The image is:
1. Converted to RGB format (if not already)
2. Resized to 30x30 pixels (model input size)
3. Normalized (pixel values from 0-255 to 0-1)
4. Shaped into batch format for the model

### Step 3: Prediction
The preprocessed image is fed to the CNN model which:
1. Extracts features through convolutional layers
2. Reduces dimensions through pooling layers
3. Makes final classification through dense layers
4. Outputs 43 probability scores (one per class)

### Step 4: Display Results
The class with the highest probability is shown as the prediction, along with its confidence score and alternative predictions.

---

## 🎓 Learning Resources

- **Dataset**: German Traffic Sign Recognition Benchmark (GTSRB)
- **Model Type**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras

---

Made with ❤️ for Road Safety
