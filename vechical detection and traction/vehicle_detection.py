# ==============================================================================
# VEHICLE DETECTION AND TRACKING SYSTEM
# ==============================================================================
# This script detects and tracks vehicles in real-time using:
# 1. Haar Cascade Classifier for detection
# 2. Centroid-based tracking for following vehicles
# 3. OpenCV for video processing and visualization
# ==============================================================================

# ==============================================================================
# SECTION 1: IMPORT LIBRARIES
# ==============================================================================

import cv2
# Line: import cv2
# What: Imports the OpenCV (Open Source Computer Vision) library
# Why: OpenCV provides all the tools we need for:
#      - Capturing video from camera or file
#      - Loading Haar Cascade classifier
#      - Drawing rectangles and text on frames
#      - Displaying the output window
# How: cv2 is the Python binding for OpenCV written in C++ for speed

import numpy as np
# Line: import numpy as np
# What: Imports NumPy library with alias 'np'
# Why: NumPy is needed for:
#      - Efficient array operations
#      - Mathematical calculations (distance between centroids)
#      - Working with image data (images are NumPy arrays)
# How: NumPy provides optimized numerical operations in Python

from collections import OrderedDict
# Line: from collections import OrderedDict
# What: Imports OrderedDict class from Python's collections module
# Why: OrderedDict maintains insertion order of items
#      We use it to store tracked vehicles with their IDs
#      This ensures consistent ordering when matching vehicles
# How: OrderedDict is a dictionary subclass that remembers insertion order

import time
# Line: import time
# What: Imports Python's time module
# Why: Used to calculate FPS (Frames Per Second)
#      by measuring time between frames
# How: time.time() returns current time in seconds

import math
# Line: import math
# What: Imports Python's math module
# Why: Used for mathematical functions like:
#      - math.sqrt() for Euclidean distance calculation
#      - math.hypot() as alternative for distance
# How: Provides access to mathematical functions

# ==============================================================================
# SECTION 2: CENTROID TRACKER CLASS
# ==============================================================================
# This class handles tracking vehicles across multiple frames
# by using the centroid (center point) of each detected vehicle

class CentroidTracker:
    # Line: class CentroidTracker:
    # What: Defines a new class called CentroidTracker
    # Why: Encapsulates all tracking logic in one organized unit
    #      Classes help us group related data and functions together
    # How: Python class definition using 'class' keyword

    def __init__(self, max_disappeared=50):
        # Line: def __init__(self, max_disappeared=50):
        # What: Constructor method - runs when creating a new CentroidTracker object
        # Why: Initializes all the variables the tracker needs
        # How: __init__ is Python's special constructor method
        #      max_disappeared=50 means if vehicle not seen for 50 frames, remove it

        self.next_object_id = 0
        # Line: self.next_object_id = 0
        # What: Counter for assigning unique IDs to vehicles
        # Why: Each vehicle needs a unique ID to track it
        #      Starts at 0, increments for each new vehicle
        # How: self.variable stores data that belongs to this object

        self.objects = OrderedDict()
        # Line: self.objects = OrderedDict()
        # What: Dictionary to store currently tracked vehicles
        # Why: Maps vehicle ID -> centroid position (x, y)
        #      Example: {0: (100, 200), 1: (300, 400)}
        # How: OrderedDict maintains order of vehicle IDs

        self.disappeared = OrderedDict()
        # Line: self.disappeared = OrderedDict()
        # What: Dictionary to count frames since vehicle was last seen
        # Why: If vehicle disappears too long, we stop tracking it
        #      Example: {0: 5, 1: 0} means ID 0 not seen for 5 frames
        # How: Maps vehicle ID -> disappearance frame count

        self.max_disappeared = max_disappeared
        # Line: self.max_disappeared = max_disappeared
        # What: Maximum frames a vehicle can be missing before removal
        # Why: Prevents ghost vehicles from staying forever
        #      After this many frames without detection, vehicle is removed
        # How: Default is 50 frames (about 1-2 seconds at 30 FPS)

        self.total_count = 0
        # Line: self.total_count = 0
        # What: Tracks TOTAL number of unique vehicles detected
        # Why: User requested to count all vehicles that have been detected
        #      This is different from current vehicles - it never decreases
        # How: Incremented each time a new vehicle is registered

    def register(self, centroid):
        # Line: def register(self, centroid):
        # What: Method to register a new vehicle for tracking
        # Why: When a new vehicle appears, we add it to our tracking system
        # How: Takes the centroid (center point) of the new vehicle

        self.objects[self.next_object_id] = centroid
        # Line: self.objects[self.next_object_id] = centroid
        # What: Stores the centroid with the current ID
        # Why: Creates the initial record for this vehicle
        # How: Adds key-value pair to objects dictionary

        self.disappeared[self.next_object_id] = 0
        # Line: self.disappeared[self.next_object_id] = 0
        # What: Initializes disappeared counter to 0
        # Why: Vehicle was just seen, so 0 frames have passed
        # How: Adds entry to disappeared dictionary

        self.next_object_id += 1
        # Line: self.next_object_id += 1
        # What: Increments the ID counter for next vehicle
        # Why: Ensures next vehicle gets a unique ID
        # How: Adds 1 to the counter

        self.total_count += 1
        # Line: self.total_count += 1
        # What: Increments the total vehicle count
        # Why: Tracks total unique vehicles detected overall
        # How: This number never decreases

    def deregister(self, object_id):
        # Line: def deregister(self, object_id):
        # What: Method to remove a vehicle from tracking
        # Why: Called when vehicle hasn't been seen for too long
        # How: Takes the ID of the vehicle to remove

        del self.objects[object_id]
        # Line: del self.objects[object_id]
        # What: Removes the vehicle from objects dictionary
        # Why: We no longer need to track this vehicle
        # How: Python's del statement removes dictionary entry

        del self.disappeared[object_id]
        # Line: del self.disappeared[object_id]
        # What: Removes the disappeared counter for this vehicle
        # Why: No need to track disappearance of removed vehicle
        # How: Removes entry from disappeared dictionary

    def update(self, rects):
        # Line: def update(self, rects):
        # What: Main method to update tracker with new detections
        # Why: Called every frame with new bounding box detections
        #      This is the heart of the tracking algorithm
        # How: Takes list of rectangles (bounding boxes)

        # ----- CASE 1: No detections in current frame -----
        if len(rects) == 0:
            # Line: if len(rects) == 0:
            # What: Checks if no vehicles were detected this frame
            # Why: If nothing detected, we mark all tracked vehicles as disappeared
            # How: len() returns number of items in the list

            for object_id in list(self.disappeared.keys()):
                # Line: for object_id in list(self.disappeared.keys()):
                # What: Loop through all currently tracked vehicle IDs
                # Why: Need to increment disappeared count for each
                # How: list() creates a copy to safely modify during iteration

                self.disappeared[object_id] += 1
                # Line: self.disappeared[object_id] += 1
                # What: Increment the disappeared counter
                # Why: Vehicle not seen this frame, so increase count
                # How: Adds 1 to the counter

                if self.disappeared[object_id] > self.max_disappeared:
                    # Line: if self.disappeared[object_id] > self.max_disappeared:
                    # What: Check if vehicle has been gone too long
                    # Why: Remove vehicles that have left the scene
                    # How: Compares counter to maximum threshold

                    self.deregister(object_id)
                    # Line: self.deregister(object_id)
                    # What: Remove this vehicle from tracking
                    # Why: It has been gone for too many frames
                    # How: Calls our deregister method

            return self.objects
            # Line: return self.objects
            # What: Return the current tracked objects
            # Why: Even with no detections, we return what we're tracking
            # How: Returns the dictionary of ID -> centroid

        # ----- Calculate centroids for all detected rectangles -----
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        # Line: input_centroids = np.zeros((len(rects), 2), dtype="int")
        # What: Creates an empty array to store centroids
        # Why: We'll calculate center point for each bounding box
        # How: Creates array with shape (num_rectangles, 2) for (x, y) pairs

        for (i, (x, y, w, h)) in enumerate(rects):
            # Line: for (i, (x, y, w, h)) in enumerate(rects):
            # What: Loop through each rectangle with its index
            # Why: Need to calculate centroid for each detection
            # How: enumerate gives (index, value), unpacking (x, y, w, h)
            #      x, y = top-left corner; w, h = width, height

            cX = int(x + w / 2.0)
            # Line: cX = int(x + w / 2.0)
            # What: Calculate center X coordinate
            # Why: Centroid X = left edge + half the width
            # How: x is left edge, w is width, so x + w/2 = center

            cY = int(y + h / 2.0)
            # Line: cY = int(y + h / 2.0)
            # What: Calculate center Y coordinate
            # Why: Centroid Y = top edge + half the height
            # How: y is top edge, h is height, so y + h/2 = center

            input_centroids[i] = (cX, cY)
            # Line: input_centroids[i] = (cX, cY)
            # What: Store the calculated centroid in our array
            # Why: We'll use these centroids for matching
            # How: Assigns the tuple to position i in the array

        # ----- CASE 2: No existing tracked objects -----
        if len(self.objects) == 0:
            # Line: if len(self.objects) == 0:
            # What: Check if we're not tracking any vehicles yet
            # Why: If nothing tracked, register all new detections
            # How: len() of empty dictionary is 0

            for i in range(len(input_centroids)):
                # Line: for i in range(len(input_centroids)):
                # What: Loop through all detected centroids
                # Why: Register each one as a new vehicle
                # How: range(n) gives 0, 1, 2, ..., n-1

                self.register(input_centroids[i])
                # Line: self.register(input_centroids[i])
                # What: Register this centroid as a new vehicle
                # Why: No existing vehicles, so all are new
                # How: Calls our register method

        # ----- CASE 3: Match existing objects with new detections -----
        else:
            # Line: else:
            # What: We have existing tracked objects AND new detections
            # Why: Need to match new detections to existing vehicles
            # How: This is the complex matching logic

            object_ids = list(self.objects.keys())
            # Line: object_ids = list(self.objects.keys())
            # What: Get list of all current vehicle IDs
            # Why: Need to compare each tracked vehicle with detections
            # How: Creates a list from dictionary keys

            object_centroids = list(self.objects.values())
            # Line: object_centroids = list(self.objects.values())
            # What: Get list of all current vehicle centroids
            # Why: Need positions to calculate distances
            # How: Creates a list from dictionary values

            # ----- Calculate distance matrix -----
            D = np.zeros((len(object_centroids), len(input_centroids)))
            # Line: D = np.zeros((len(object_centroids), len(input_centroids)))
            # What: Create distance matrix (rows=existing, cols=new)
            # Why: We need distance from each old to each new centroid
            # How: Matrix where D[i][j] = distance from object i to input j

            for i in range(len(object_centroids)):
                # Line: for i in range(len(object_centroids)):
                # What: Loop through each existing tracked vehicle
                # Why: Calculate distances to all new detections
                # How: i is the index of the tracked vehicle

                for j in range(len(input_centroids)):
                    # Line: for j in range(len(input_centroids)):
                    # What: Loop through each new detection
                    # Why: Calculate distance from vehicle i to detection j
                    # How: j is the index of the new detection

                    D[i, j] = math.dist(object_centroids[i], input_centroids[j])
                    # Line: D[i, j] = math.dist(object_centroids[i], input_centroids[j])
                    # What: Calculate Euclidean distance between two points
                    # Why: Smaller distance = likely same vehicle
                    # How: math.dist computes sqrt((x2-x1)² + (y2-y1)²)

            # ----- Find optimal matching using greedy algorithm -----
            rows = D.min(axis=1).argsort()
            # Line: rows = D.min(axis=1).argsort()
            # What: Sort existing objects by their closest match distance
            # Why: Process objects with closest matches first (greedy approach)
            # How: min(axis=1) finds min distance per row, argsort() gives sorted indices

            cols = D.argmin(axis=1)[rows]
            # Line: cols = D.argmin(axis=1)[rows]
            # What: Get the column index (new detection) of minimum distance for each row
            # Why: Tells us which new detection is closest to each existing object
            # How: argmin(axis=1) returns index of min value in each row

            used_rows = set()
            # Line: used_rows = set()
            # What: Set to track which existing objects have been matched
            # Why: Each object should match at most one detection
            # How: set() provides O(1) lookup and uniqueness

            used_cols = set()
            # Line: used_cols = set()
            # What: Set to track which new detections have been matched
            # Why: Each detection should match at most one object
            # How: Prevents double-matching

            for (row, col) in zip(rows, cols):
                # Line: for (row, col) in zip(rows, cols):
                # What: Loop through pairs of (object_index, detection_index)
                # Why: Process matches in order of closest distance
                # How: zip combines two lists into pairs

                if row in used_rows or col in used_cols:
                    # Line: if row in used_rows or col in used_cols:
                    # What: Check if either has already been matched
                    # Why: Skip if already used to prevent double-matching
                    # How: O(1) lookup in sets

                    continue
                    # Line: continue
                    # What: Skip to next iteration of the loop
                    # Why: Don't process already-matched items
                    # How: Python's continue statement

                object_id = object_ids[row]
                # Line: object_id = object_ids[row]
                # What: Get the ID of the existing vehicle
                # Why: Need the ID to update its position
                # How: Look up ID from our list using the row index

                self.objects[object_id] = input_centroids[col]
                # Line: self.objects[object_id] = input_centroids[col]
                # What: Update the vehicle's position to new centroid
                # Why: Vehicle has moved, update its tracked position
                # How: Overwrites the old centroid with the new one

                self.disappeared[object_id] = 0
                # Line: self.disappeared[object_id] = 0
                # What: Reset disappeared counter to 0
                # Why: Vehicle was seen this frame
                # How: Sets the counter back to 0

                used_rows.add(row)
                # Line: used_rows.add(row)
                # What: Mark this existing object as matched
                # Why: Prevent it from being matched again
                # How: Adds to the set

                used_cols.add(col)
                # Line: used_cols.add(col)
                # What: Mark this new detection as matched
                # Why: Prevent it from being matched again
                # How: Adds to the set

            # ----- Handle unmatched items -----
            unused_rows = set(range(len(object_centroids))) - used_rows
            # Line: unused_rows = set(range(len(object_centroids))) - used_rows
            # What: Find existing objects that weren't matched
            # Why: These vehicles may have disappeared
            # How: Set difference: all rows minus matched rows

            unused_cols = set(range(len(input_centroids))) - used_cols
            # Line: unused_cols = set(range(len(input_centroids))) - used_cols
            # What: Find new detections that weren't matched
            # Why: These are potentially new vehicles
            # How: Set difference: all columns minus matched columns

            # ----- Handle disappeared objects -----
            for row in unused_rows:
                # Line: for row in unused_rows:
                # What: Loop through unmatched existing objects
                # Why: Increment their disappeared counters
                # How: Iterates over the set of unused row indices

                object_id = object_ids[row]
                # Line: object_id = object_ids[row]
                # What: Get the ID of the unmatched vehicle
                # Why: Need ID to update disappeared counter
                # How: Look up from our list

                self.disappeared[object_id] += 1
                # Line: self.disappeared[object_id] += 1
                # What: Increment disappeared counter
                # Why: Vehicle not detected this frame
                # How: Adds 1 to the counter

                if self.disappeared[object_id] > self.max_disappeared:
                    # Line: if self.disappeared[object_id] > self.max_disappeared:
                    # What: Check if gone for too long
                    # Why: Remove if exceeded threshold
                    # How: Compare to max value

                    self.deregister(object_id)
                    # Line: self.deregister(object_id)
                    # What: Remove from tracking
                    # Why: Vehicle has left the scene
                    # How: Calls deregister method

            # ----- Register new objects -----
            for col in unused_cols:
                # Line: for col in unused_cols:
                # What: Loop through unmatched new detections
                # Why: These are new vehicles entering the scene
                # How: Iterates over unused column indices

                self.register(input_centroids[col])
                # Line: self.register(input_centroids[col])
                # What: Register as a new tracked vehicle
                # Why: New vehicle detected
                # How: Calls register method

        return self.objects
        # Line: return self.objects
        # What: Return all currently tracked objects
        # Why: Caller needs to know tracked vehicles and positions
        # How: Returns dictionary of ID -> centroid

# ==============================================================================
# SECTION 3: MAIN FUNCTION
# ==============================================================================

def main():
    # Line: def main():
    # What: Defines the main entry point function
    # Why: Organizes the main program logic in one function
    # How: Standard Python convention for main function

    # ----- Configuration -----
    CASCADE_FILE = "cars.xml"
    # Line: CASCADE_FILE = "cars.xml"
    # What: Path to the Haar Cascade classifier file
    # Why: This XML file contains the trained model for car detection
    # How: String containing the filename (same directory as script)

    VIDEO_SOURCE = 0
    # Line: VIDEO_SOURCE = 0
    # What: Video source for capture (0 = default webcam)
    # Why: 0 refers to the primary camera
    #      You can change this to a video file path like "traffic.mp4"
    # How: Integer for camera index OR string for file path

    MIN_AREA = 1000
    # Line: MIN_AREA = 1000
    # What: Minimum area (in pixels²) for a detection to be valid
    # Why: Filters out small false positive detections
    #      Real vehicles should have larger bounding boxes
    # How: Area = width * height of bounding box

    SCALE_FACTOR = 1.1
    # Line: SCALE_FACTOR = 1.1
    # What: Scale factor for multi-scale detection
    # Why: Cascade classifier scans image at multiple sizes
    #      1.1 means 10% size increase at each scale
    #      Smaller = more accurate but slower
    # How: Used by detectMultiScale function

    MIN_NEIGHBORS = 3
    # Line: MIN_NEIGHBORS = 3
    # What: Minimum number of neighbor rectangles for valid detection
    # Why: Higher value = fewer false positives but may miss some
    #      Lower value = more detections but more false positives
    # How: Controls detection strictness

    # ----- Load the Haar Cascade Classifier -----
    print("[INFO] Loading Haar Cascade classifier...")
    # Line: print("[INFO] Loading Haar Cascade classifier...")
    # What: Prints status message to console
    # Why: Informs user that loading is starting
    # How: print() outputs text to terminal

    car_cascade = cv2.CascadeClassifier(CASCADE_FILE)
    # Line: car_cascade = cv2.CascadeClassifier(CASCADE_FILE)
    # What: Creates a CascadeClassifier object from the XML file
    # Why: This loads the pre-trained model for car detection
    #      The model contains features learned from thousands of car images
    # How: OpenCV parses the XML and creates the classifier

    if car_cascade.empty():
        # Line: if car_cascade.empty():
        # What: Checks if the classifier failed to load
        # Why: File may not exist or be corrupted
        # How: empty() returns True if loading failed

        print(f"[ERROR] Could not load cascade file: {CASCADE_FILE}")
        # Line: print(f"[ERROR] Could not load cascade file: {CASCADE_FILE}")
        # What: Prints error message with filename
        # Why: Helps user understand what went wrong
        # How: f-string allows variable insertion in string

        print("[ERROR] Make sure 'cars.xml' is in the same folder as this script.")
        # Line: print("[ERROR] Make sure 'cars.xml' is in the same folder as this script.")
        # What: Provides solution hint to user
        # Why: Most common fix for this error
        # How: Simple print statement

        return
        # Line: return
        # What: Exit the function early
        # Why: Cannot continue without the classifier
        # How: Ends function execution

    print("[INFO] Cascade classifier loaded successfully!")
    # Line: print("[INFO] Cascade classifier loaded successfully!")
    # What: Confirms successful loading
    # Why: User knows the classifier is ready
    # How: Status message to console

    # ----- Initialize Video Capture -----
    print(f"[INFO] Starting video capture from source: {VIDEO_SOURCE}")
    # Line: print(f"[INFO] Starting video capture from source: {VIDEO_SOURCE}")
    # What: Prints video source information
    # Why: User knows what source is being used
    # How: f-string includes the source value

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    # Line: cap = cv2.VideoCapture(VIDEO_SOURCE)
    # What: Creates a VideoCapture object
    # Why: This object handles reading video frames
    #      Works with webcam (index) or video file (path)
    # How: OpenCV's video capture interface

    if not cap.isOpened():
        # Line: if not cap.isOpened():
        # What: Checks if video source opened successfully
        # Why: Camera may be in use or file may not exist
        # How: isOpened() returns True if capture is ready

        print("[ERROR] Could not open video source!")
        # Line: print("[ERROR] Could not open video source!")
        # What: Prints error message
        # Why: Informs user of the failure
        # How: Simple print statement

        print("[TIP] Try changing VIDEO_SOURCE to a video file path")
        # Line: print("[TIP] Try changing VIDEO_SOURCE to a video file path")
        # What: Provides suggestion for fixing the issue
        # Why: Webcam issues are common; video file is alternative
        # How: Helpful tip for user

        return
        # Line: return
        # What: Exit the function early
        # Why: Cannot continue without video source
        # How: Ends function execution

    print("[INFO] Video capture started successfully!")
    # Line: print("[INFO] Video capture started successfully!")
    # What: Confirms video capture is working
    # Why: User knows video source is ready
    # How: Status message

    # ----- Initialize Tracker -----
    tracker = CentroidTracker(max_disappeared=40)
    # Line: tracker = CentroidTracker(max_disappeared=40)
    # What: Creates an instance of our CentroidTracker class
    # Why: This object will track vehicles across frames
    #      max_disappeared=40 means remove after ~1.3 seconds at 30 FPS
    # How: Calls the constructor of CentroidTracker

    print("[INFO] Centroid tracker initialized")
    # Line: print("[INFO] Centroid tracker initialized")
    # What: Confirms tracker is ready
    # Why: Status update for user
    # How: Simple print statement

    # ----- Setup Display Window -----
    window_name = "Vehicle Detection and Tracking"
    # Line: window_name = "Vehicle Detection and Tracking"
    # What: Name for the display window
    # Why: Identifies the window and appears in title bar
    # How: String that will be used by OpenCV window functions

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Line: cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # What: Creates a named window with resizable property
    # Why: WINDOW_NORMAL allows user to resize the window
    #      WINDOW_AUTOSIZE would lock the size
    # How: OpenCV window creation function

    cv2.resizeWindow(window_name, 1280, 720)
    # Line: cv2.resizeWindow(window_name, 1280, 720)
    # What: Sets initial window size to 1280x720 (720p)
    # Why: Good default size for viewing
    # How: Width=1280 pixels, Height=720 pixels

    # ----- Variables for FPS calculation -----
    fps = 0
    # Line: fps = 0
    # What: Variable to store current FPS value
    # Why: We'll display this on screen
    # How: Initialized to 0, updated each frame

    prev_time = time.time()
    # Line: prev_time = time.time()
    # What: Records the current time
    # Why: Need previous time to calculate time difference
    # How: time.time() returns seconds since epoch

    fps_update_counter = 0
    # Line: fps_update_counter = 0
    # What: Counter to reduce FPS display update frequency
    # Why: Updating FPS every frame causes flickering
    # How: Only update FPS display every N frames

    print("[INFO] Starting detection loop...")
    # Line: print("[INFO] Starting detection loop...")
    # What: Indicates main loop is starting
    # Why: User knows processing is beginning
    # How: Status message

    print("[INFO] Press 'q' to quit")
    # Line: print("[INFO] Press 'q' to quit")
    # What: Instructs user how to exit
    # Why: User needs to know the exit key
    # How: 'q' key will trigger exit

    # ==============================================================================
    # SECTION 4: MAIN DETECTION LOOP
    # ==============================================================================

    while True:
        # Line: while True:
        # What: Infinite loop that processes frames continuously
        # Why: Runs until user presses 'q' to quit
        # How: while True creates a loop that never ends on its own

        # ----- Read a frame from video -----
        ret, frame = cap.read()
        # Line: ret, frame = cap.read()
        # What: Captures a single frame from the video source
        # Why: Need frames to process for detection
        # How: ret = success boolean, frame = image as NumPy array
        #      ret is True if frame was read successfully

        if not ret:
            # Line: if not ret:
            # What: Check if frame reading failed
            # Why: Video may have ended or camera disconnected
            # How: ret is False when no frame is available

            print("[INFO] End of video or camera disconnected")
            # Line: print("[INFO] End of video or camera disconnected")
            # What: Inform user why loop is ending
            # Why: Could be normal end or error
            # How: Status message

            break
            # Line: break
            # What: Exit the while loop
            # Why: No more frames to process
            # How: Python's break statement exits the loop

        # ----- Calculate FPS -----
        current_time = time.time()
        # Line: current_time = time.time()
        # What: Get current time for FPS calculation
        # Why: Compare with previous time to get elapsed time
        # How: Returns current time in seconds

        fps_update_counter += 1
        # Line: fps_update_counter += 1
        # What: Increment the FPS update counter
        # Why: We'll update FPS every 10 frames
        # How: Adds 1 to counter

        if fps_update_counter >= 10:
            # Line: if fps_update_counter >= 10:
            # What: Check if we should update FPS display
            # Why: Updating every frame causes flickering
            # How: Every 10 frames gives smoother display

            elapsed = current_time - prev_time
            # Line: elapsed = current_time - prev_time
            # What: Calculate time passed since last update
            # Why: FPS = frames / time
            # How: Simple subtraction of times

            fps = fps_update_counter / elapsed if elapsed > 0 else 0
            # Line: fps = fps_update_counter / elapsed if elapsed > 0 else 0
            # What: Calculate frames per second
            # Why: Shows performance metric to user
            # How: FPS = number of frames / time elapsed
            #      Ternary operator prevents division by zero

            prev_time = current_time
            # Line: prev_time = current_time
            # What: Update previous time for next calculation
            # Why: Need reference point for next FPS calculation
            # How: Store current time for comparison

            fps_update_counter = 0
            # Line: fps_update_counter = 0
            # What: Reset the counter
            # Why: Start counting frames for next update
            # How: Sets back to 0

        # ----- Convert frame to grayscale -----
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Line: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # What: Converts the color frame to grayscale
        # Why: Haar Cascade works on grayscale images
        #      Grayscale is faster to process (1 channel vs 3)
        #      BGR = Blue, Green, Red (OpenCV's color format)
        # How: cvtColor converts between color spaces

        # ----- Detect vehicles using Haar Cascade -----
        vehicles = car_cascade.detectMultiScale(
            gray,                    # Input grayscale image
            scaleFactor=SCALE_FACTOR,  # Scale factor for multi-scale detection
            minNeighbors=MIN_NEIGHBORS, # Minimum neighbors for valid detection
            minSize=(30, 30)         # Minimum object size
        )
        # Line: vehicles = car_cascade.detectMultiScale(...)
        # What: Detects vehicles in the grayscale image
        # Why: This is the core detection step using Haar features
        # How: Returns list of rectangles (x, y, width, height)
        #
        # Parameters explained:
        # - gray: Input image (must be grayscale)
        # - scaleFactor: How much to reduce image size at each scale
        #   1.1 means reduce by 10% each step
        # - minNeighbors: How many neighbors each rectangle needs
        #   Higher = less false positives but may miss some
        # - minSize: Minimum object size to detect (30x30 pixels)

        # ----- Filter detections by area -----
        rects = []
        # Line: rects = []
        # What: Empty list to store valid detections
        # Why: We'll filter out small/invalid detections
        # How: Start with empty list, append valid ones

        for (x, y, w, h) in vehicles:
            # Line: for (x, y, w, h) in vehicles:
            # What: Loop through each detected vehicle
            # Why: Process and filter each detection
            # How: Unpack tuple: x,y = top-left corner, w,h = size

            area = w * h
            # Line: area = w * h
            # What: Calculate area of bounding box
            # Why: Filter out small false positive detections
            # How: Area = width × height

            if area >= MIN_AREA:
                # Line: if area >= MIN_AREA:
                # What: Check if detection is large enough
                # Why: Small detections are likely false positives
                # How: Compare with threshold (1000 pixels²)

                rects.append((x, y, w, h))
                # Line: rects.append((x, y, w, h))
                # What: Add valid detection to our list
                # Why: Only keep detections meeting criteria
                # How: Append tuple to list

        # ----- Update tracker with detections -----
        objects = tracker.update(rects)
        # Line: objects = tracker.update(rects)
        # What: Update the tracker with new detections
        # Why: Matches new detections with existing tracked objects
        #      Assigns IDs and tracks movement
        # How: Returns dictionary of ID -> centroid

        # ----- Draw bounding boxes and IDs -----
        for (x, y, w, h) in rects:
            # Line: for (x, y, w, h) in rects:
            # What: Loop through valid detections
            # Why: Draw bounding box for each
            # How: Iterate over filtered rectangle list

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Line: cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # What: Draws a rectangle on the frame
            # Why: Visualize the detected vehicle area
            # How: Parameters:
            #      - frame: Image to draw on
            #      - (x, y): Top-left corner
            #      - (x + w, y + h): Bottom-right corner
            #      - (0, 255, 0): Color in BGR (Green)
            #      - 2: Line thickness in pixels

        # ----- Draw vehicle IDs and centroids -----
        for (object_id, centroid) in objects.items():
            # Line: for (object_id, centroid) in objects.items():
            # What: Loop through tracked objects
            # Why: Draw ID label for each tracked vehicle
            # How: items() returns (key, value) pairs

            text = f"ID {object_id}"
            # Line: text = f"ID {object_id}"
            # What: Create text label with vehicle ID
            # Why: Shows which vehicle is which
            # How: f-string formats the ID number

            cv2.putText(frame, text, (centroid[0] - 20, centroid[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # Line: cv2.putText(frame, text, ...)
            # What: Draws text on the frame
            # Why: Display the vehicle ID near the vehicle
            # How: Parameters:
            #      - frame: Image to draw on
            #      - text: The string to display
            #      - (centroid[0] - 20, centroid[1] - 20): Position (offset up-left)
            #      - cv2.FONT_HERSHEY_SIMPLEX: Font type
            #      - 0.6: Font scale
            #      - (0, 0, 255): Color in BGR (Red)
            #      - 2: Line thickness

            cv2.circle(frame, (centroid[0], centroid[1]), 5, (255, 0, 0), -1)
            # Line: cv2.circle(frame, (centroid[0], centroid[1]), 5, (255, 0, 0), -1)
            # What: Draws a filled circle at the centroid
            # Why: Shows exact center point of tracked vehicle
            # How: Parameters:
            #      - frame: Image to draw on
            #      - (centroid[0], centroid[1]): Center point
            #      - 5: Radius in pixels
            #      - (255, 0, 0): Color in BGR (Blue)
            #      - -1: Thickness (-1 means filled)

        # ----- Draw information panel -----
        # Background rectangle for text readability
        cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)
        # Line: cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)
        # What: Draws a filled black rectangle
        # Why: Creates dark background for text readability
        # How: -1 thickness means filled rectangle

        cv2.rectangle(frame, (10, 10), (350, 120), (0, 255, 0), 2)
        # Line: cv2.rectangle(frame, (10, 10), (350, 120), (0, 255, 0), 2)
        # What: Draws green border around the panel
        # Why: Makes the info panel stand out
        # How: 2 pixel thick green border

        # Display current vehicle count
        current_count = len(objects)
        # Line: current_count = len(objects)
        # What: Count of currently tracked vehicles
        # Why: Shows how many vehicles are in frame right now
        # How: len() of objects dictionary

        cv2.putText(frame, f"Current Vehicles: {current_count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Line: cv2.putText(frame, f"Current Vehicles: {current_count}", ...)
        # What: Displays current vehicle count
        # Why: User sees live count of vehicles in frame
        # How: Green text at top of info panel

        # Display total vehicle count
        cv2.putText(frame, f"Total Vehicles: {tracker.total_count}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # Line: cv2.putText(frame, f"Total Vehicles: {tracker.total_count}", ...)
        # What: Displays total unique vehicles detected
        # Why: User requested total count feature
        #      This shows all vehicles that have ever been detected
        # How: Yellow text (0, 255, 255 = Cyan) below current count

        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Line: cv2.putText(frame, f"FPS: {fps:.1f}", ...)
        # What: Displays frames per second
        # Why: Shows performance/processing speed
        # How: White text, .1f formats to 1 decimal place

        # ----- Display the frame -----
        cv2.imshow(window_name, frame)
        # Line: cv2.imshow(window_name, frame)
        # What: Displays the processed frame in the window
        # Why: Shows the result to the user
        # How: Updates the window with new frame content

        # ----- Check for quit key -----
        key = cv2.waitKey(1) & 0xFF
        # Line: key = cv2.waitKey(1) & 0xFF
        # What: Waits 1ms for a key press, gets the key code
        # Why: Need to check if user pressed 'q' to quit
        #      Also allows the window to update
        # How: waitKey(1) waits 1ms, & 0xFF masks to get ASCII value

        if key == ord('q'):
            # Line: if key == ord('q'):
            # What: Check if the pressed key is 'q'
            # Why: 'q' is our designated quit key
            # How: ord('q') returns ASCII value of 'q' (113)

            print("[INFO] Quit key pressed. Exiting...")
            # Line: print("[INFO] Quit key pressed. Exiting...")
            # What: Inform user of exit
            # Why: Confirmation of intentional exit
            # How: Status message

            break
            # Line: break
            # What: Exit the while loop
            # Why: User wants to quit
            # How: Breaks out of the infinite loop

    # ==============================================================================
    # SECTION 5: CLEANUP
    # ==============================================================================

    print("\n" + "=" * 50)
    # Line: print("\n" + "=" * 50)
    # What: Prints a separator line
    # Why: Visual separation for final summary
    # How: \n = newline, "=" * 50 repeats "=" 50 times

    print("DETECTION SUMMARY")
    # Line: print("DETECTION SUMMARY")
    # What: Header for final summary
    # Why: Clearly labels the summary section
    # How: Simple print statement

    print("=" * 50)
    # Line: print("=" * 50)
    # What: Prints another separator line
    # Why: Visual consistency
    # How: Repeats "=" 50 times

    print(f"Total unique vehicles detected: {tracker.total_count}")
    # Line: print(f"Total unique vehicles detected: {tracker.total_count}")
    # What: Prints the final total count
    # Why: Gives user the complete count for the session
    # How: f-string with the total_count variable

    print("=" * 50)
    # Line: print("=" * 50)
    # What: Final separator line
    # Why: Clean visual ending
    # How: Same pattern

    cap.release()
    # Line: cap.release()
    # What: Releases the video capture resource
    # Why: Frees the camera/file for other applications
    #      Important for proper resource management
    # How: release() closes the video capture

    cv2.destroyAllWindows()
    # Line: cv2.destroyAllWindows()
    # What: Closes all OpenCV windows
    # Why: Clean shutdown, removes windows from screen
    # How: Destroys all windows created by OpenCV

    print("[INFO] Cleanup complete. Goodbye!")
    # Line: print("[INFO] Cleanup complete. Goodbye!")
    # What: Final confirmation message
    # Why: User knows program ended cleanly
    # How: Friendly goodbye message

# ==============================================================================
# SECTION 6: PROGRAM ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Line: if __name__ == "__main__":
    # What: Checks if this script is being run directly
    # Why: Allows the script to be imported without running main()
    #      When run directly, __name__ equals "__main__"
    #      When imported, __name__ equals the module name
    # How: Python's standard idiom for executable scripts

    main()
    # Line: main()
    # What: Calls the main function to start the program
    # Why: Executes all the detection and tracking logic
    # How: Function call with no arguments
