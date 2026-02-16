import cv2
import numpy as np

def clean_nothing(x):
    pass

def main():
    cap = cv2.VideoCapture(0)
    
    cv2.namedWindow("Calibration")
    
    # Create Trackbars
    # Hue is 0-179, Sat/Val are 0-255
    cv2.createTrackbar("H Min", "Calibration", 0, 179, clean_nothing)
    cv2.createTrackbar("S Min", "Calibration", 0, 255, clean_nothing)
    cv2.createTrackbar("V Min", "Calibration", 0, 255, clean_nothing)
    
    cv2.createTrackbar("H Max", "Calibration", 179, 179, clean_nothing)
    cv2.createTrackbar("S Max", "Calibration", 255, 255, clean_nothing)
    cv2.createTrackbar("V Max", "Calibration", 255, 255, clean_nothing)
    
    # Set default values for generic skin color (just a starting point)
    cv2.setTrackbarPos("H Min", "Calibration", 0)
    cv2.setTrackbarPos("S Min", "Calibration", 20)
    cv2.setTrackbarPos("V Min", "Calibration", 70)
    cv2.setTrackbarPos("H Max", "Calibration", 20)
    cv2.setTrackbarPos("S Max", "Calibration", 255)
    cv2.setTrackbarPos("V Max", "Calibration", 255)

    print("Use the sliders to isolate your hand so it appears WHITE and background BLACK.")
    print("Press 's' to save and print the values.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        h_min = cv2.getTrackbarPos("H Min", "Calibration")
        s_min = cv2.getTrackbarPos("S Min", "Calibration")
        v_min = cv2.getTrackbarPos("V Min", "Calibration")
        
        h_max = cv2.getTrackbarPos("H Max", "Calibration")
        s_max = cv2.getTrackbarPos("S Max", "Calibration")
        v_max = cv2.getTrackbarPos("V Max", "Calibration")
        
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # Simple noise filtering visualization
        mask_result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Stack images for easier viewing
        # Resize for fitting on screen if needed
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        stacked = np.hstack([frame, mask_result, mask_bgr])
        
        # Resize to fit screen width roughly
        scale = 0.6
        h, w = stacked.shape[:2]
        stacked = cv2.resize(stacked, (int(w*scale), int(h*scale)))
        
        cv2.imshow("Calibration", stacked)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(f"\nSaved Values:\nLower: {lower}\nUpper: {upper}")
            # We could save to a file, but printing is fine for now as per plan
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
