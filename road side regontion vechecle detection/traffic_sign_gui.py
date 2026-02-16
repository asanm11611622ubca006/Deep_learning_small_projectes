"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PROFESSIONAL TRAFFIC SIGN RECOGNITION GUI                   ║
║                          Roadside Vehicle Detection System                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

This application provides a modern, professional interface for traffic sign
recognition using a pre-trained CNN model. Features include:
- Drag-and-drop image upload
- Real-time prediction with confidence scores
- Beautiful dark-themed UI with glassmorphism effects
- Support for multiple image formats
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os
import sys

# TensorFlow import with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
except ImportError:
    messagebox.showerror("Error", "TensorFlow is not installed. Please run: pip install tensorflow")
    sys.exit(1)


# ============================================================================
# TRAFFIC SIGN CLASSES (German Traffic Sign Recognition Dataset - 43 classes)
# ============================================================================
TRAFFIC_SIGN_CLASSES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5t",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5t"
}


class ModernButton(tk.Canvas):
    """Custom modern button with gradient and hover effects"""
    
    def __init__(self, parent, text, command=None, width=200, height=50, 
                 bg_color="#667eea", hover_color="#764ba2", **kwargs):
        super().__init__(parent, width=width, height=height, 
                        highlightthickness=0, bg=parent["bg"], **kwargs)
        
        self.text = text
        self.command = command
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.current_color = bg_color
        
        self.draw_button()
        
        # Bind events
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
    
    def draw_button(self):
        self.delete("all")
        
        # Draw rounded rectangle
        radius = 12
        self.create_rounded_rect(2, 2, self.width-2, self.height-2, 
                                radius, fill=self.current_color, outline="")
        
        # Draw text
        self.create_text(self.width//2, self.height//2, text=self.text,
                        fill="white", font=("Segoe UI", 12, "bold"))
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1+radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def _on_enter(self, event):
        self.current_color = self.hover_color
        self.draw_button()
    
    def _on_leave(self, event):
        self.current_color = self.bg_color
        self.draw_button()
    
    def _on_click(self, event):
        if self.command:
            self.command()


class TrafficSignRecognitionApp:
    """
    Main application class for the Traffic Sign Recognition GUI.
    Features a modern, dark-themed interface with glassmorphism effects.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Roadside Traffic Sign Recognition System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        self.root.configure(bg="#0f0f23")
        
        # Variables
        self.model = None
        self.current_image_path = None
        self.prediction_result = None
        
        # Load model
        self.load_model()
        
        # Setup UI
        self.setup_styles()
        self.create_widgets()
        
    def setup_styles(self):
        """Configure custom styles for the application"""
        self.colors = {
            "bg_primary": "#0f0f23",
            "bg_secondary": "#1a1a2e",
            "bg_tertiary": "#16213e",
            "accent": "#667eea",
            "accent_hover": "#764ba2",
            "success": "#00d9a5",
            "warning": "#ffc107",
            "error": "#ff6b6b",
            "text_primary": "#ffffff",
            "text_secondary": "#a0a0b0",
            "border": "#2a2a4a"
        }
        
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use("clam")
        
        # Progress bar style
        style.configure("Custom.Horizontal.TProgressbar",
                       background=self.colors["accent"],
                       troughcolor=self.colors["bg_secondary"],
                       borderwidth=0,
                       lightcolor=self.colors["accent"],
                       darkcolor=self.colors["accent"])
    
    def load_model(self):
        """Load the pre-trained model"""
        model_path = os.path.join(os.path.dirname(__file__), "my_model.h5")
        
        try:
            self.model = keras.models.load_model(model_path)
            print(f"[SUCCESS] Model loaded from: {model_path}")
        except Exception as e:
            messagebox.showerror("Model Error", 
                               f"Failed to load model: {str(e)}\n\nPlease ensure 'my_model.h5' exists in the application directory.")
            self.model = None
    
    def create_widgets(self):
        """Create all UI widgets"""
        
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors["bg_primary"])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # =====================================================================
        # HEADER SECTION
        # =====================================================================
        header_frame = tk.Frame(main_container, bg=self.colors["bg_primary"])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Title
        title_label = tk.Label(header_frame, 
                              text="🚦 Traffic Sign Recognition",
                              font=("Segoe UI", 28, "bold"),
                              fg=self.colors["text_primary"],
                              bg=self.colors["bg_primary"])
        title_label.pack(side=tk.LEFT)
        
        # Subtitle
        subtitle_label = tk.Label(header_frame,
                                 text="Roadside Detection System • AI Powered",
                                 font=("Segoe UI", 12),
                                 fg=self.colors["text_secondary"],
                                 bg=self.colors["bg_primary"])
        subtitle_label.pack(side=tk.LEFT, padx=(20, 0), pady=(10, 0))
        
        # Model status indicator
        status_text = "● Model Loaded" if self.model else "○ Model Not Loaded"
        status_color = self.colors["success"] if self.model else self.colors["error"]
        status_label = tk.Label(header_frame,
                               text=status_text,
                               font=("Segoe UI", 11),
                               fg=status_color,
                               bg=self.colors["bg_primary"])
        status_label.pack(side=tk.RIGHT, pady=(10, 0))
        
        # =====================================================================
        # CONTENT SECTION
        # =====================================================================
        content_frame = tk.Frame(main_container, bg=self.colors["bg_primary"])
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Image upload area
        self.create_upload_panel(content_frame)
        
        # Right panel - Results area
        self.create_results_panel(content_frame)
        
        # =====================================================================
        # FOOTER SECTION
        # =====================================================================
        footer_frame = tk.Frame(main_container, bg=self.colors["bg_primary"])
        footer_frame.pack(fill=tk.X, pady=(20, 0))
        
        footer_text = "© 2026 Roadside Traffic Sign Recognition System • 43 Sign Classes Supported"
        footer_label = tk.Label(footer_frame,
                               text=footer_text,
                               font=("Segoe UI", 9),
                               fg=self.colors["text_secondary"],
                               bg=self.colors["bg_primary"])
        footer_label.pack(side=tk.LEFT)
    
    def create_upload_panel(self, parent):
        """Create the image upload panel"""
        
        # Left panel container
        left_panel = tk.Frame(parent, bg=self.colors["bg_secondary"], 
                             highlightbackground=self.colors["border"],
                             highlightthickness=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Panel header
        panel_header = tk.Frame(left_panel, bg=self.colors["bg_tertiary"])
        panel_header.pack(fill=tk.X)
        
        header_label = tk.Label(panel_header,
                               text="📤 Upload Image",
                               font=("Segoe UI", 14, "bold"),
                               fg=self.colors["text_primary"],
                               bg=self.colors["bg_tertiary"],
                               pady=12)
        header_label.pack(side=tk.LEFT, padx=15)
        
        # Image display area
        image_container = tk.Frame(left_panel, bg=self.colors["bg_secondary"])
        image_container.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # Drop zone
        self.drop_zone = tk.Frame(image_container, 
                                 bg=self.colors["bg_tertiary"],
                                 highlightbackground=self.colors["accent"],
                                 highlightthickness=2)
        self.drop_zone.pack(expand=True, padx=20)
        
        # Image label (for displaying uploaded image)
        self.image_label = tk.Label(self.drop_zone, 
                                   bg=self.colors["bg_tertiary"],
                                   cursor="hand2")
        self.image_label.pack(expand=True, pady=30, padx=30)
        
        # Placeholder content
        self.show_placeholder()
        
        # Button container
        button_container = tk.Frame(left_panel, bg=self.colors["bg_secondary"])
        button_container.pack(fill=tk.X, pady=(0, 20), padx=20)
        
        # Upload button
        upload_btn = ModernButton(button_container, "📁 Select Image", 
                                 command=self.browse_image, width=180, height=45)
        upload_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear button
        clear_btn = ModernButton(button_container, "🗑️ Clear", 
                                command=self.clear_image, width=120, height=45,
                                bg_color="#4a4a6a", hover_color="#5a5a7a")
        clear_btn.pack(side=tk.LEFT)
        
        # Bind click event on drop zone
        self.drop_zone.bind("<Button-1>", lambda e: self.browse_image())
        self.image_label.bind("<Button-1>", lambda e: self.browse_image())
    
    def create_results_panel(self, parent):
        """Create the results panel"""
        
        # Right panel container
        right_panel = tk.Frame(parent, bg=self.colors["bg_secondary"],
                              highlightbackground=self.colors["border"],
                              highlightthickness=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Panel header
        panel_header = tk.Frame(right_panel, bg=self.colors["bg_tertiary"])
        panel_header.pack(fill=tk.X)
        
        header_label = tk.Label(panel_header,
                               text="🎯 Prediction Results",
                               font=("Segoe UI", 14, "bold"),
                               fg=self.colors["text_primary"],
                               bg=self.colors["bg_tertiary"],
                               pady=12)
        header_label.pack(side=tk.LEFT, padx=15)
        
        # Results container
        results_container = tk.Frame(right_panel, bg=self.colors["bg_secondary"])
        results_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Main prediction result
        self.result_frame = tk.Frame(results_container, bg=self.colors["bg_tertiary"],
                                    highlightbackground=self.colors["border"],
                                    highlightthickness=1)
        self.result_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Prediction label
        self.prediction_label = tk.Label(self.result_frame,
                                        text="Upload an image to see prediction",
                                        font=("Segoe UI", 16),
                                        fg=self.colors["text_secondary"],
                                        bg=self.colors["bg_tertiary"],
                                        wraplength=400,
                                        pady=25)
        self.prediction_label.pack()
        
        # Confidence section
        confidence_frame = tk.Frame(results_container, bg=self.colors["bg_tertiary"],
                                   highlightbackground=self.colors["border"],
                                   highlightthickness=1)
        confidence_frame.pack(fill=tk.X, pady=(0, 20))
        
        confidence_header = tk.Label(confidence_frame,
                                    text="Confidence Score",
                                    font=("Segoe UI", 12, "bold"),
                                    fg=self.colors["text_primary"],
                                    bg=self.colors["bg_tertiary"])
        confidence_header.pack(anchor=tk.W, padx=15, pady=(15, 5))
        
        # Progress bar for confidence
        self.confidence_bar = ttk.Progressbar(confidence_frame, 
                                             style="Custom.Horizontal.TProgressbar",
                                             length=350, mode="determinate")
        self.confidence_bar.pack(padx=15, pady=(0, 5))
        
        # Confidence percentage label
        self.confidence_label = tk.Label(confidence_frame,
                                        text="0%",
                                        font=("Segoe UI", 14, "bold"),
                                        fg=self.colors["accent"],
                                        bg=self.colors["bg_tertiary"])
        self.confidence_label.pack(anchor=tk.E, padx=15, pady=(0, 15))
        
        # Top 5 predictions
        top5_frame = tk.Frame(results_container, bg=self.colors["bg_tertiary"],
                             highlightbackground=self.colors["border"],
                             highlightthickness=1)
        top5_frame.pack(fill=tk.BOTH, expand=True)
        
        top5_header = tk.Label(top5_frame,
                              text="Top 5 Predictions",
                              font=("Segoe UI", 12, "bold"),
                              fg=self.colors["text_primary"],
                              bg=self.colors["bg_tertiary"])
        top5_header.pack(anchor=tk.W, padx=15, pady=(15, 10))
        
        # Create labels for top 5 predictions
        self.top5_labels = []
        self.top5_bars = []
        
        for i in range(5):
            row_frame = tk.Frame(top5_frame, bg=self.colors["bg_tertiary"])
            row_frame.pack(fill=tk.X, padx=15, pady=3)
            
            rank_label = tk.Label(row_frame,
                                 text=f"#{i+1}",
                                 font=("Segoe UI", 10, "bold"),
                                 fg=self.colors["accent"],
                                 bg=self.colors["bg_tertiary"],
                                 width=3)
            rank_label.pack(side=tk.LEFT)
            
            name_label = tk.Label(row_frame,
                                 text="---",
                                 font=("Segoe UI", 10),
                                 fg=self.colors["text_secondary"],
                                 bg=self.colors["bg_tertiary"],
                                 anchor=tk.W,
                                 width=35)
            name_label.pack(side=tk.LEFT, padx=(5, 0))
            
            conf_label = tk.Label(row_frame,
                                 text="0%",
                                 font=("Segoe UI", 10, "bold"),
                                 fg=self.colors["text_secondary"],
                                 bg=self.colors["bg_tertiary"],
                                 width=8)
            conf_label.pack(side=tk.RIGHT)
            
            self.top5_labels.append((name_label, conf_label))
        
        # Predict button
        predict_container = tk.Frame(right_panel, bg=self.colors["bg_secondary"])
        predict_container.pack(fill=tk.X, pady=(0, 20), padx=20)
        
        predict_btn = ModernButton(predict_container, "🔍 Predict Sign", 
                                  command=self.predict_image, width=200, height=50,
                                  bg_color="#00d9a5", hover_color="#00b389")
        predict_btn.pack()
    
    def show_placeholder(self):
        """Show placeholder content in the drop zone"""
        
        # Clear previous content
        for widget in self.image_label.winfo_children():
            widget.destroy()
        
        # Create placeholder
        placeholder_frame = tk.Frame(self.image_label, bg=self.colors["bg_tertiary"])
        placeholder_frame.pack(expand=True)
        
        icon_label = tk.Label(placeholder_frame,
                             text="🖼️",
                             font=("Segoe UI", 48),
                             bg=self.colors["bg_tertiary"])
        icon_label.pack()
        
        text_label = tk.Label(placeholder_frame,
                             text="Click to upload image\nor drag and drop here",
                             font=("Segoe UI", 12),
                             fg=self.colors["text_secondary"],
                             bg=self.colors["bg_tertiary"],
                             justify=tk.CENTER)
        text_label.pack(pady=(10, 0))
        
        format_label = tk.Label(placeholder_frame,
                               text="Supports: JPG, PNG, BMP, GIF",
                               font=("Segoe UI", 9),
                               fg=self.colors["text_secondary"],
                               bg=self.colors["bg_tertiary"])
        format_label.pack(pady=(5, 0))
        
        self.image_label.configure(image="")
    
    def browse_image(self):
        """Open file dialog to select an image"""
        
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Traffic Sign Image",
            filetypes=filetypes
        )
        
        if filepath:
            self.load_image(filepath)
    
    def load_image(self, filepath):
        """Load and display the selected image"""
        
        try:
            # Open and resize image for display
            img = Image.open(filepath)
            
            # Calculate display size (max 350x350 while maintaining aspect ratio)
            max_size = 350
            ratio = min(max_size / img.width, max_size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            
            display_img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Add subtle shadow effect
            shadow = Image.new('RGBA', (new_size[0] + 10, new_size[1] + 10), (0, 0, 0, 0))
            shadow.paste((30, 30, 50, 100), (5, 5, new_size[0] + 5, new_size[1] + 5))
            shadow = shadow.filter(ImageFilter.GaussianBlur(5))
            
            # Convert to PhotoImage
            self.display_photo = ImageTk.PhotoImage(display_img)
            
            # Clear placeholder and show image
            for widget in self.image_label.winfo_children():
                widget.destroy()
            
            self.image_label.configure(image=self.display_photo)
            self.current_image_path = filepath
            
            # Auto-predict
            self.predict_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def clear_image(self):
        """Clear the current image and results"""
        
        self.current_image_path = None
        self.show_placeholder()
        
        # Reset results
        self.prediction_label.configure(
            text="Upload an image to see prediction",
            fg=self.colors["text_secondary"]
        )
        self.confidence_bar["value"] = 0
        self.confidence_label.configure(text="0%")
        
        for name_label, conf_label in self.top5_labels:
            name_label.configure(text="---")
            conf_label.configure(text="0%")
    
    def preprocess_image(self, image_path):
        """
        Preprocess the image for model prediction.
        
        The model expects:
        - Image size: 30x30 pixels
        - Color: RGB (3 channels)
        - Normalized pixel values: 0-1 range
        """
        
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to 30x30 (model input size)
        img = img.resize((30, 30), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize pixel values to 0-1 range
        img_array = img_array / 255.0
        
        # Add batch dimension (model expects shape: (batch_size, 30, 30, 3))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_image(self):
        """Perform prediction on the current image"""
        
        if not self.model:
            messagebox.showwarning("Warning", "Model not loaded. Cannot perform prediction.")
            return
        
        if not self.current_image_path:
            messagebox.showinfo("Info", "Please upload an image first.")
            return
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(self.current_image_path)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get predicted class and confidence
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100
            
            # Get class name
            class_name = TRAFFIC_SIGN_CLASSES.get(predicted_class, f"Unknown Class {predicted_class}")
            
            # Update main prediction
            self.prediction_label.configure(
                text=f"🎯 {class_name}",
                fg=self.colors["success"] if confidence > 70 else self.colors["warning"]
            )
            
            # Update confidence bar
            self.confidence_bar["value"] = confidence
            self.confidence_label.configure(
                text=f"{confidence:.1f}%",
                fg=self.colors["success"] if confidence > 70 else 
                   self.colors["warning"] if confidence > 40 else self.colors["error"]
            )
            
            # Update top 5 predictions
            top5_indices = np.argsort(predictions[0])[-5:][::-1]
            
            for i, idx in enumerate(top5_indices):
                class_name = TRAFFIC_SIGN_CLASSES.get(idx, f"Class {idx}")
                conf = predictions[0][idx] * 100
                
                name_label, conf_label = self.top5_labels[i]
                name_label.configure(
                    text=class_name[:40] + "..." if len(class_name) > 40 else class_name,
                    fg=self.colors["text_primary"] if i == 0 else self.colors["text_secondary"]
                )
                conf_label.configure(
                    text=f"{conf:.1f}%",
                    fg=self.colors["success"] if i == 0 else self.colors["text_secondary"]
                )
            
            self.prediction_result = {
                "class_id": int(predicted_class),
                "class_name": class_name,
                "confidence": float(confidence),
                "all_predictions": predictions[0].tolist()
            }
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed to predict: {str(e)}")
            print(f"Error details: {e}")


def main():
    """Main entry point for the application"""
    
    root = tk.Tk()
    
    # Set window icon (if available)
    try:
        # You can add a custom icon here
        pass
    except:
        pass
    
    # Create application
    app = TrafficSignRecognitionApp(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"+{x}+{y}")
    
    # Run application
    root.mainloop()


if __name__ == "__main__":
    main()
