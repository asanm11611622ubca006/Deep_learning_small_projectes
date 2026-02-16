import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import json
import os

# Parameters
IMG_HEIGHT = 28
IMG_WIDTH = 28
MODEL_PATH = 'character_cnn.h5'
JSON_PATH = 'class_indices.json'
TRAIN_DIR = 'train'

class CharacterRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Character Recognition System")
        self.root.geometry("550x700")
        self.root.resizable(False, False)

        # Load Model
        self.model = None
        self.class_indices = {}
        self.load_model_and_labels()

        # UI Components
        self.create_widgets()

    def load_model_and_labels(self):
        # 1. Load Model
        try:
            if os.path.exists(MODEL_PATH):
                self.model = tf.keras.models.load_model(MODEL_PATH)
                print("Model loaded successfully.")
            else:
                messagebox.showerror("Error", f"Model file '{MODEL_PATH}' not found.\nPlease train the model first.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")

        # 2. Load Class Indices
        if os.path.exists(JSON_PATH):
            try:
                with open(JSON_PATH, 'r') as f:
                    self.class_indices = json.load(f)
                    self.class_indices = {int(k): v for k, v in self.class_indices.items()}
                print("Labels loaded from JSON.")
            except Exception as e:
                print(f"Error loading JSON: {e}")
        
        if not self.class_indices and os.path.exists(TRAIN_DIR):
            print("Inferring labels from 'train' directory...")
            try:
                classes = sorted(os.listdir(TRAIN_DIR))
                self.class_indices = {i: label for i, label in enumerate(classes)}
                print("Labels inferred from directory structure.")
            except Exception as e:
                print(f"Error checking train dir: {e}")
        
        if not self.class_indices:
             print("Warning: No labels found.")

    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="Character Recognition", font=("Helvetica", 24, "bold"))
        title_label.pack(pady=20)

        # Image Display Area
        self.image_label = tk.Label(self.root, text="No Image Uploaded", bg="gray", width=40, height=15)
        self.image_label.pack(pady=10)

        # Processed Image Display (shows what the model sees)
        proc_frame = tk.Frame(self.root)
        proc_frame.pack(pady=5)
        tk.Label(proc_frame, text="Model Input (28x28):", font=("Arial", 10)).pack(side=tk.LEFT)
        self.processed_label = tk.Label(proc_frame, bg="black", width=8, height=4)
        self.processed_label.pack(side=tk.LEFT, padx=10)

        # Prediction Result
        self.result_label = tk.Label(self.root, text="Prediction: None", font=("Helvetica", 20, "bold"), fg="blue")
        self.result_label.pack(pady=15)

        # Top 3 predictions
        self.top3_label = tk.Label(self.root, text="", font=("Arial", 10), fg="gray")
        self.top3_label.pack(pady=5)

        # Controls Frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=20)

        # Upload Button
        btn_upload = tk.Button(control_frame, text="Upload Image", command=self.upload_image, 
                               font=("Arial", 12), bg="#4CAF50", fg="white", padx=20, pady=10)
        btn_upload.pack(side=tk.LEFT, padx=10)

        # Exit Button
        btn_exit = tk.Button(control_frame, text="Exit", command=self.root.quit, 
                             font=("Arial", 12), bg="#f44336", fg="white", padx=20, pady=10)
        btn_exit.pack(side=tk.RIGHT, padx=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.process_and_predict(file_path)

    def process_and_predict(self, file_path):
        try:
            # 1. Display Original Image
            pil_img_display = Image.open(file_path)
            pil_img_display.thumbnail((300, 300)) 
            self.tk_image = ImageTk.PhotoImage(pil_img_display)
            self.image_label.config(image=self.tk_image, text="", width=300, height=300)

            if self.model is None:
                self.result_label.config(text="Error: Model not loaded")
                return

            # 2. Preprocess EXACTLY like ImageDataGenerator does during training
            # Use Keras image loading to match training pipeline exactly
            img = keras_image.load_img(
                file_path,
                color_mode='grayscale',  # Convert to grayscale
                target_size=(IMG_HEIGHT, IMG_WIDTH)  # Resize to 28x28
            )
            
            # Convert to numpy array
            img_array = keras_image.img_to_array(img)  # Shape: (28, 28, 1)
            
            # Normalize to 0-1 (same as rescale=1./255 in ImageDataGenerator)
            img_array = img_array / 255.0
            
            # Add batch dimension: (1, 28, 28, 1)
            input_data = np.expand_dims(img_array, axis=0)

            # Show what the model sees (scaled up for visibility)
            processed_display = Image.fromarray((img_array[:,:,0] * 255).astype(np.uint8))
            processed_display = processed_display.resize((56, 56), Image.Resampling.NEAREST)
            self.tk_processed = ImageTk.PhotoImage(processed_display)
            self.processed_label.config(image=self.tk_processed)

            # 3. Predict
            prediction = self.model.predict(input_data, verbose=0)
            predicted_class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            # Get label
            label = self.class_indices.get(predicted_class_index, str(predicted_class_index))
            
            # Display Result
            self.result_label.config(text=f"Prediction: {label} ({confidence*100:.1f}%)")
            
            # Show top 3 predictions
            top3_indices = np.argsort(prediction[0])[-3:][::-1]
            top3_text = "Top 3: " + ", ".join([
                f"{self.class_indices.get(idx, str(idx))} ({prediction[0][idx]*100:.1f}%)" 
                for idx in top3_indices
            ])
            self.top3_label.config(text=top3_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = CharacterRecognitionApp(root)
    root.mainloop()
