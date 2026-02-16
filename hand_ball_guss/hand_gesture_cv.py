import cv2
import numpy as np
import math

# ==========================================
# CONFIGURATION
# ==========================================
# REPLACE THESE WITH YOUR VALUES FROM calibrate_hand.py
# Standard Skin Color (Lighting Dependent!)
LOWER_HSV = np.array([0, 20, 70])
UPPER_HSV = np.array([20, 255, 255])

# Smoothing factor for circle movement (0.0 - 1.0)
ALPHA = 0.2
# ==========================================

def get_palm_center(contour, shape):
    """
    Approximates palm center using max enclosed circle (Distance Transform)
    or simply centroid. Distance transform is more robust for 'palm'.
    """
    # Create a mask for the contour
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Distance Transform: finding the point furthest from any background pixel
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist)
    
    # maxLoc is the center of the palm (circle with max radius fits here)
    radius = int(maxVal)
    return maxLoc, radius

def main():
    cap = cv2.VideoCapture(0)
    
    # State variables for smoothing
    smooth_x, smooth_y = 0, 0
    smooth_radius = 50
    
    print("Starting Hand Gesture Control...")
    print("Ensure you have good lighting and the background is distinct from your hand.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Flip & Preprocessing
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        
        # 2. Color Segmentation
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        
        # Morphology to remove noise (Erosion -> Dilation)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2) # Dilate more to fill holes

        # 3. Contour Detection
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest contour (The Hand)
        max_cnt = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_cnt = cnt

        # Process if hand is detected (Area threshold to avoid noise)
        if max_cnt is not None and max_area > 3000:
            # 4. Hull & Defects
            hull = cv2.convexHull(max_cnt)
            
            # Find Palm Center
            palm_center, palm_radius = get_palm_center(max_cnt, frame.shape)
            cx, cy = palm_center
            
            # Draw Palm Center (Debugging)
            # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1) 
            
            # 5. Find Fingertips
            # We look for points on the hull that are far from the center
            # Simplify hull indices for convexity defects
            hull_indices = cv2.convexHull(max_cnt, returnPoints=False)
            
            try:
                defects = cv2.convexityDefects(max_cnt, hull_indices)
            except Exception:
                defects = None

            # Get potential finger tips from Hull
            # Heuristic: Hull points that are local peaks in distance from center
            # Simple approach: All hull points are candidates, filter by distance from palm center
            
            finger_points = []
            if defects is not None:
                # Use defects to find deep valleys between fingers, 
                # but we actually want the peaks (fingertips).
                # Hull points are generally the fingertips.
                for point in hull:
                    px, py = point[0]
                    dist = math.hypot(px - cx, py - cy)
                    
                    # If point is outside the palm circle * factor, it's likely a finger
                    if dist > palm_radius * 1.5:
                        finger_points.append((px, py))
            
            # Reduce cluster of points (several hull points might be on one fingertip)
            # We just take the ones that are far apart
            final_fingers = []
            for fp in finger_points:
                is_distinct = True
                for ff in final_fingers:
                    if math.hypot(fp[0]-ff[0], fp[1]-ff[1]) < 30: # 30px threshold
                        is_distinct = False
                        break
                if is_distinct:
                    final_fingers.append(fp)
            
            # Filter top 2 furthest points from center (Simulating Index + Thumb/Pinky)
            # Sort by distance from center descending
            final_fingers.sort(key=lambda p: math.hypot(p[0]-cx, p[1]-cy), reverse=True)
            
            active_points = final_fingers[:2]
            
            # DRAWING LOGIC
            target_radius = 40 # Default
            
            # If we have at least 1 point, follow it
            if len(active_points) >= 1:
                # Primary point (e.g., Index)
                ix, iy = active_points[0]
                
                # Update smooth position
                smooth_x = int(smooth_x + (ix - smooth_x) * ALPHA)
                smooth_y = int(smooth_y + (iy - smooth_y) * ALPHA)
                
                # If we have 2 points, distance controls radius
                if len(active_points) >= 2:
                    tx, ty = active_points[1]
                    dist = math.hypot(tx - ix, ty - iy)
                    
                    # Map distance: 50px -> 250px map to Radius 40 -> 120
                    target_radius = int(np.interp(dist, [50, 250], [40, 120]))
                    
                    # Draw connection line
                    cv2.line(frame, active_points[0], active_points[1], (0, 255, 0), 2)
                    for p in active_points:
                        cv2.circle(frame, p, 8, (0, 0, 255), -1)

            # Smooth Radius change
            smooth_radius = int(smooth_radius + (target_radius - smooth_radius) * 0.1)

            # Draw the Main Visual (Similar to user request)
            # 1. Dark backing
            cv2.circle(frame, (smooth_x+8, smooth_y+8), smooth_radius+6, (20, 20, 20), -1)

            # 2. Radiating rings
            for r in range(smooth_radius+20, smooth_radius, -5):
                overlay = frame.copy()
                cv2.circle(overlay, (smooth_x, smooth_y), r, (255, 120, 0), 2)
                alpha_ring = 0.4 - ((r - smooth_radius) / 100.0) # Fade out
                if alpha_ring < 0: alpha_ring = 0
                frame = cv2.addWeighted(overlay, alpha_ring, frame, 1 - alpha_ring, 0)

            # 3. Main circles
            cv2.circle(frame, (smooth_x, smooth_y), smooth_radius, (255, 160, 20), -1)
            cv2.circle(frame, (smooth_x, smooth_y), int(smooth_radius*0.6), (255, 220, 120), -1)
            
            # Draw contours for feedback (Optional, keeping it subtle)
            cv2.drawContours(frame, [max_cnt], -1, (0, 255, 0), 1)
        
        else:
            # If no hand found, text hint
            cv2.putText(frame, "No hand detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("AI Hand Controlled 3D Object (CV Approach)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
