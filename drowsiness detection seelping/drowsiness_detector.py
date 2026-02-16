"""
=============================================================================
EYE DROWSINESS DETECTION SYSTEM USING DLIB
=============================================================================

This advanced Python script detects drowsiness by monitoring eye closure using
the Eye Aspect Ratio (EAR) algorithm with dlib's 68-point facial landmarks.

Key Features:
- Real-time face detection using OpenCV's Haar Cascade
- Eye landmark detection using dlib's 68-point shape predictor
- Eye Aspect Ratio (EAR) calculation for accurate drowsiness detection
- Beep sound alert when drowsiness is detected
- Visual feedback with eye landmarks and status display

Algorithm: Eye Aspect Ratio (EAR)
---------------------------------
The EAR is computed using 6 landmark points around each eye:

        P2    P3
     P1          P4
        P6    P5

    EAR = (||P2 - P6|| + ||P3 - P5||) / (2 * ||P1 - P4||)

- Open eyes: EAR ≈ 0.25 to 0.35
- Closed eyes: EAR < 0.20

Author: AI Assistant
Date: January 2026
=============================================================================
"""

# =============================================================================
# SECTION 1: IMPORT REQUIRED LIBRARIES
# =============================================================================

import cv2                      # OpenCV for video capture and image processing
import dlib                     # dlib for facial landmark prediction
import numpy as np              # NumPy for numerical operations
from scipy.spatial import distance as dist  # For Euclidean distance calculation
import winsound                 # Windows-specific module for beep sound
import time                     # For time-related operations
from collections import OrderedDict  # For ordered dictionary of facial landmarks
import os                       # For file path operations

# =============================================================================
# SECTION 2: CONFIGURATION PARAMETERS
# =============================================================================

# Eye Aspect Ratio threshold - below this value, eyes are considered closed
# Typical values: 0.20 - 0.30 (adjust based on individual eye shape)
EAR_THRESHOLD = 0.25

# Number of consecutive frames the eye must be below threshold to trigger alert
# At 30 FPS, 30 frames = 1 second of closed eyes
CONSECUTIVE_FRAMES_THRESHOLD = 30

# Beep sound parameters (Windows winsound)
BEEP_FREQUENCY = 2500  # Frequency in Hz (higher = higher pitch)
BEEP_DURATION = 500    # Duration in milliseconds

# Path to dlib's pre-trained facial landmark predictor
# Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# =============================================================================
# SECTION 3: FACIAL LANDMARKS INDICES
# =============================================================================

"""
dlib's 68-point facial landmark detector provides points for:
- Jaw: 0-16
- Right Eyebrow: 17-21
- Left Eyebrow: 22-26
- Nose: 27-35
- Right Eye: 36-41 (6 points)
- Left Eye: 42-47 (6 points)
- Mouth: 48-67

For drowsiness detection, we focus on the eye landmarks:
"""

# Define indices for left and right eyes in the 68-point model
# These are 0-indexed, so right eye is points 36-41, left eye is points 42-47
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("right_eye", (36, 42)),   # Right eye: landmarks 36 to 41 (exclusive 42)
    ("left_eye", (42, 48)),    # Left eye: landmarks 42 to 47 (exclusive 48)
])

# Extract the start and end indices for eyes
(RIGHT_EYE_START, RIGHT_EYE_END) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
(LEFT_EYE_START, LEFT_EYE_END) = FACIAL_LANDMARKS_68_IDXS["left_eye"]

# =============================================================================
# SECTION 4: EYE ASPECT RATIO (EAR) CALCULATION
# =============================================================================

def calculate_eye_aspect_ratio(eye):
    """
    Calculate the Eye Aspect Ratio (EAR) for a given eye.
    
    The EAR is a scalar value that represents whether the eye is open or closed.
    It's calculated using 6 landmark points around the eye.
    
    Eye landmarks configuration:
    
            P2 (1)    P3 (2)
         P1 (0)          P4 (3)
            P6 (5)    P5 (4)
    
    Formula:
        EAR = (||P2 - P6|| + ||P3 - P5||) / (2 * ||P1 - P4||)
    
    Where ||.|| denotes the Euclidean distance between two points.
    
    Parameters:
    -----------
    eye : numpy.ndarray
        Array of 6 (x, y) coordinates representing the eye landmarks
        Shape: (6, 2)
    
    Returns:
    --------
    float
        The Eye Aspect Ratio value
        - Higher values (~0.25-0.35) indicate open eyes
        - Lower values (<0.20) indicate closed eyes
    
    Mathematical Explanation:
    -------------------------
    - The numerator computes the vertical distances (heights) of the eye
    - The denominator computes the horizontal distance (width) of the eye
    - When eyes close, vertical distances decrease while horizontal stays same
    - This causes EAR to decrease, allowing us to detect eye closure
    """
    
    # Compute the Euclidean distances between the two sets of vertical eye landmarks
    # These represent the "height" of the eye opening
    
    # Vertical distance between P2 (index 1) and P6 (index 5)
    vertical_dist_1 = dist.euclidean(eye[1], eye[5])
    
    # Vertical distance between P3 (index 2) and P5 (index 4)
    vertical_dist_2 = dist.euclidean(eye[2], eye[4])
    
    # Compute the Euclidean distance between the horizontal eye landmarks
    # This represents the "width" of the eye
    
    # Horizontal distance between P1 (index 0) and P4 (index 3)
    horizontal_dist = dist.euclidean(eye[0], eye[3])
    
    # Calculate the Eye Aspect Ratio
    # Average of vertical distances divided by horizontal distance
    # Multiplied by 2 in denominator to normalize since we have 2 vertical measurements
    ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
    
    return ear

# =============================================================================
# SECTION 5: FACIAL LANDMARK CONVERSION
# =============================================================================

def shape_to_numpy(shape, dtype="int"):
    """
    Convert dlib's shape object to a NumPy array.
    
    dlib returns facial landmarks as a special 'shape' object.
    This function converts it to a NumPy array for easier manipulation.
    
    Parameters:
    -----------
    shape : dlib.full_object_detection
        The dlib shape object containing facial landmarks
    dtype : str
        Data type for the output array (default: "int")
    
    Returns:
    --------
    numpy.ndarray
        Array of shape (68, 2) containing (x, y) coordinates
        for all 68 facial landmarks
    """
    
    # Initialize an empty array to store the coordinates
    # Shape: (68 points, 2 coordinates per point)
    coords = np.zeros((68, 2), dtype=dtype)
    
    # Loop through all 68 landmarks
    for i in range(68):
        # Extract x and y coordinates for each landmark point
        coords[i] = (shape.part(i).x, shape.part(i).y)
    
    return coords

# =============================================================================
# SECTION 6: VISUALIZATION HELPER FUNCTIONS
# =============================================================================

def draw_eye_contour(frame, eye_points, color=(0, 255, 0)):
    """
    Draw the contour around detected eye landmarks.
    
    Parameters:
    -----------
    frame : numpy.ndarray
        The video frame to draw on
    eye_points : numpy.ndarray
        Array of 6 (x, y) coordinates for eye landmarks
    color : tuple
        BGR color for the contour (default: green)
    """
    
    # Convert points to the format required by cv2.convexHull
    eye_hull = cv2.convexHull(eye_points)
    
    # Draw the convex hull around the eye
    cv2.drawContours(frame, [eye_hull], -1, color, 2)

def draw_status_display(frame, ear, status, frame_counter, fps):
    """
    Draw status information overlay on the video frame.
    
    Parameters:
    -----------
    frame : numpy.ndarray
        The video frame to draw on
    ear : float
        Current Eye Aspect Ratio value
    status : str
        Current drowsiness status ("AWAKE" or "DROWSY")
    frame_counter : int
        Number of consecutive frames with closed eyes
    fps : float
        Current frames per second
    """
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Draw semi-transparent status bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Display EAR value
    ear_text = f"EAR: {ear:.3f}"
    cv2.putText(frame, ear_text, (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Display frame counter
    counter_text = f"Drowsy Frames: {frame_counter}/{CONSECUTIVE_FRAMES_THRESHOLD}"
    cv2.putText(frame, counter_text, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # Display FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (width - 120, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # Display status with appropriate color
    if status == "DROWSY":
        # Red warning for drowsy state
        status_color = (0, 0, 255)
        status_text = "!! DROWSY! WAKE UP! !!"
        
        # Draw flashing warning rectangle
        cv2.rectangle(frame, (0, height - 80), (width, height), (0, 0, 200), -1)
        cv2.putText(frame, status_text, (width // 2 - 180, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    else:
        # Green for awake state
        status_color = (0, 255, 0)
        cv2.putText(frame, "Status: AWAKE", (width - 200, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

# =============================================================================
# SECTION 7: ALERT SYSTEM
# =============================================================================

def play_alert_sound():
    """
    Play a beep sound to alert the user.
    
    Uses Windows winsound module for audio playback.
    The beep is non-blocking to avoid pausing video processing.
    """
    
    try:
        # Play beep sound with specified frequency and duration
        # This is Windows-specific
        winsound.Beep(BEEP_FREQUENCY, BEEP_DURATION)
    except Exception as e:
        print(f"[WARNING] Could not play alert sound: {e}")

# =============================================================================
# SECTION 8: OPENCV FACE DETECTOR WRAPPER
# =============================================================================

def rect_to_dlib_rect(x, y, w, h):
    """
    Convert OpenCV rectangle (x, y, w, h) to dlib rectangle format.
    
    Parameters:
    -----------
    x, y : int
        Top-left corner coordinates
    w, h : int
        Width and height of the rectangle
    
    Returns:
    --------
    dlib.rectangle
        A dlib rectangle object
    """
    return dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

# =============================================================================
# SECTION 9: MAIN DROWSINESS DETECTION SYSTEM
# =============================================================================

class DrowsinessDetector:
    """
    Main class for the Drowsiness Detection System.
    
    This class encapsulates all functionality for detecting drowsiness
    using eye aspect ratio monitoring with dlib facial landmarks.
    Uses OpenCV's Haar Cascade for face detection and dlib for landmarks.
    
    Attributes:
    -----------
    face_cascade : cv2.CascadeClassifier
        OpenCV's Haar Cascade face detector
    landmark_predictor : dlib.shape_predictor
        68-point facial landmark predictor
    frame_counter : int
        Counter for consecutive drowsy frames
    is_drowsy : bool
        Current drowsiness state
    """
    
    def __init__(self):
        """
        Initialize the Drowsiness Detector.
        
        Loads the OpenCV face cascade and dlib facial landmark predictor.
        """
        
        print("=" * 60)
        print("    EYE DROWSINESS DETECTION SYSTEM")
        print("    Using dlib and Eye Aspect Ratio (EAR)")
        print("=" * 60)
        print()
        
        # Initialize OpenCV's Haar Cascade face detector
        # This is a fast, reliable method for frontal face detection
        print("[INFO] Loading OpenCV Haar Cascade face detector...")
        
        # Get the path to the Haar Cascade file from OpenCV's data folder
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise Exception("Could not load Haar Cascade classifier!")
        
        print("[INFO] Face detector loaded successfully!")
        
        # Initialize dlib's facial landmark predictor
        # Uses the pre-trained 68-point model
        print(f"[INFO] Loading dlib landmark predictor from: {PREDICTOR_PATH}")
        try:
            self.landmark_predictor = dlib.shape_predictor(PREDICTOR_PATH)
            print("[INFO] Landmark predictor loaded successfully!")
        except RuntimeError as e:
            print()
            print("=" * 60)
            print("[ERROR] Failed to load facial landmark predictor!")
            print()
            print("Please download the model file from:")
            print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print()
            print("Steps:")
            print("1. Download the .bz2 file")
            print("2. Extract it to get 'shape_predictor_68_face_landmarks.dat'")
            print("3. Place the .dat file in the same folder as this script")
            print("=" * 60)
            raise
        
        # Initialize frame counter for consecutive drowsy frames
        self.frame_counter = 0
        
        # Initialize drowsiness state
        self.is_drowsy = False
        
        # Time tracking for FPS calculation
        self.prev_time = time.time()
        self.fps = 0
        
        print()
        print("[INFO] Drowsiness Detector initialized successfully!")
        print(f"[INFO] EAR Threshold: {EAR_THRESHOLD}")
        print(f"[INFO] Consecutive Frames Threshold: {CONSECUTIVE_FRAMES_THRESHOLD}")
        print()
    
    def process_frame(self, frame):
        """
        Process a single video frame for drowsiness detection.
        
        This method performs the complete pipeline:
        1. Convert frame to grayscale
        2. Detect faces using Haar Cascade
        3. For each face, detect landmarks using dlib
        4. Extract eye landmarks and calculate EAR
        5. Determine drowsiness state
        6. Trigger alert if necessary
        
        Parameters:
        -----------
        frame : numpy.ndarray
            The input video frame (BGR format)
        
        Returns:
        --------
        numpy.ndarray
            The processed frame with visualizations
        """
        
        # Calculate FPS
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.prev_time + 0.0001)
        self.prev_time = current_time
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using OpenCV's Haar Cascade
        # Parameters: scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Initialize current EAR (will be updated if faces detected)
        current_ear = 0.0
        status = "AWAKE"
        
        # Convert to RGB for dlib shape predictor (required format)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Convert OpenCV rectangle to dlib rectangle format
            dlib_rect = rect_to_dlib_rect(x, y, w, h)
            
            # Get facial landmarks for this face
            # Returns 68 points outlining facial features
            # Uses RGB image as required by dlib
            landmarks = self.landmark_predictor(rgb, dlib_rect)
            
            # Convert dlib shape to numpy array for easier manipulation
            landmarks_np = shape_to_numpy(landmarks)
            
            # Extract left and right eye landmarks
            # Left eye: points 42-47, Right eye: points 36-41
            left_eye = landmarks_np[LEFT_EYE_START:LEFT_EYE_END]
            right_eye = landmarks_np[RIGHT_EYE_START:RIGHT_EYE_END]
            
            # Calculate Eye Aspect Ratio for both eyes
            left_ear = calculate_eye_aspect_ratio(left_eye)
            right_ear = calculate_eye_aspect_ratio(right_eye)
            
            # Average EAR from both eyes for more robust detection
            # This handles cases where one eye might be partially visible
            current_ear = (left_ear + right_ear) / 2.0
            
            # Draw eye contours on the frame
            # Green for normal, Red for closed
            eye_color = (0, 255, 0) if current_ear >= EAR_THRESHOLD else (0, 0, 255)
            draw_eye_contour(frame, left_eye, eye_color)
            draw_eye_contour(frame, right_eye, eye_color)
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Check if eyes are below threshold (potentially drowsy)
            if current_ear < EAR_THRESHOLD:
                # Increment drowsy frame counter
                self.frame_counter += 1
                
                # Check if we've exceeded the threshold for consecutive frames
                if self.frame_counter >= CONSECUTIVE_FRAMES_THRESHOLD:
                    status = "DROWSY"
                    
                    # Play alert sound if not already in drowsy state
                    if not self.is_drowsy:
                        self.is_drowsy = True
                        print("[ALERT] Drowsiness detected! Playing alert sound...")
                        # Play alert in a way that doesn't block video
                        play_alert_sound()
            else:
                # Eyes are open - reset counter and state
                self.frame_counter = 0
                self.is_drowsy = False
                status = "AWAKE"
        
        # Draw status display overlay
        draw_status_display(frame, current_ear, status, self.frame_counter, self.fps)
        
        # If no faces detected, show message
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return frame
    
    def run(self):
        """
        Main loop for the drowsiness detection system.
        
        Captures video from webcam, processes each frame,
        and displays the results in a window.
        
        Press 'q' to quit the application.
        """
        
        print("[INFO] Starting webcam capture...")
        print("[INFO] Press 'q' to quit")
        print()
        
        # Initialize video capture from default webcam
        # 0 = default camera, 1 = second camera, etc.
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("[ERROR] Could not open webcam!")
            print("[INFO] Please check if your webcam is connected.")
            return
        
        # Set camera resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("[INFO] Webcam started successfully!")
        print("[INFO] Position your face in front of the camera...")
        print()
        
        try:
            while True:
                # Read frame from webcam
                ret, frame = cap.read()
                
                # Check if frame was read successfully
                if not ret:
                    print("[ERROR] Failed to read frame from webcam!")
                    break
                
                # Mirror the frame horizontally for more intuitive display
                frame = cv2.flip(frame, 1)
                
                # Ensure frame is valid and properly formatted
                if frame is None or frame.size == 0:
                    continue
                
                # Process frame for drowsiness detection
                processed_frame = self.process_frame(frame)
                
                # Display the processed frame
                cv2.imshow("Eye Drowsiness Detection System", processed_frame)
                
                # Check for 'q' key press to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print()
                    print("[INFO] 'q' pressed - Exiting...")
                    break
                    
        except KeyboardInterrupt:
            print()
            print("[INFO] Interrupted by user - Exiting...")
        
        finally:
            # Clean up resources
            print("[INFO] Releasing webcam...")
            cap.release()
            
            print("[INFO] Closing windows...")
            cv2.destroyAllWindows()
            
            print("[INFO] Drowsiness Detection System stopped.")
            print()
            print("Thank you for using the Eye Drowsiness Detection System!")

# =============================================================================
# SECTION 10: MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for the application.
    
    Creates a DrowsinessDetector instance and runs the detection system.
    """
    
    print()
    print("=" * 60)
    print("       EYE DROWSINESS DETECTION SYSTEM")
    print("       Using dlib & Eye Aspect Ratio (EAR)")
    print("=" * 60)
    print()
    
    try:
        # Create and run the drowsiness detector
        detector = DrowsinessDetector()
        detector.run()
        
    except FileNotFoundError:
        print()
        print("[ERROR] Required file not found!")
        print("Please ensure 'shape_predictor_68_face_landmarks.dat' is in the current directory.")
        
    except Exception as e:
        print()
        print(f"[ERROR] An unexpected error occurred: {e}")
        print("Please check your installation and try again.")

# Entry point - only run if this script is executed directly
if __name__ == "__main__":
    main()
