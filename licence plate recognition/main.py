import cv2
import pytesseract
import numpy as np
import pandas as pd
import os

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    print(f"Processing {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None, None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Bilateral Filter to remove noise while keeping edges sharp
    # This is often better than GaussianBlur for OCR
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Canny Edge Detection
    edged = cv2.Canny(gray, 30, 200)

    return img, edged

def find_countours_and_plate(original_img, edged_img):
    # Find contours based on the edges
    contours, _ = cv2.findContours(edged_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area in descending order and take top 10
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    screenCnt = None
    
    # Loop over the contours to find a rectangular box (likely the plate)
    for c in contours:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        # If the approximated contour has 4 points, we assume we found the screen/plate
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        print("No rectangular contour detected. Attempting to use the whole image.")
        return original_img
    
    # Draw the contour on the original image for visualization (optional)
    cv2.drawContours(original_img, [screenCnt], -1, (0, 255, 0), 3)
    
    # Masking the part other than the license plate
    mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [screenCnt], 0, 255, -1)
    
    # Bitwise AND to extract the plate
    # We use the grayscale version for Tesseract, but here let's just use the original for masking
    # Actually, let's extract the ROI directly from the grayscale image?
    # Let's simple mask the grayscale image
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    new_image = cv2.bitwise_and(gray, gray, mask=mask)

    # Now we need to crop the image to the bounding box of the contour
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped = gray[topx:bottomx+1, topy:bottomy+1]

    return cropped

def main():
    # List of images to process
    image_files = ['car1.jpg', 'car2.jpg']

    # List to store results
    results = []

    for image_file in image_files:
        if not os.path.exists(image_file):
            print(f"File not found: {image_file}")
            continue

        original_img, edged_img = preprocess_image(image_file)
        if original_img is None:
            continue
            
        # Try to find the plate and crop it
        # Note: If contour detection fails, we might just pass the processed gray image or the original
        # For simplicity, let's try to extract the plate.
        plate_img = find_countours_and_plate(original_img, edged_img)
        
        # Display the Canny Edge result (optional, but good for debugging)
        # cv2.imshow("Canny Edges", edged_img)
        # cv2.waitKey(0) 

        # Configuration for Tesseract
        # psm 6: Assume a single uniform block of text.
        # psm 7: Treat the image as a single text line.
        custom_config = r'--oem 3 --psm 7' 
        
        text = pytesseract.image_to_string(plate_img, config=custom_config)
        detected_text = text.strip()
        
        print(f"--- Results for {image_file} ---")
        print(f"Detected Text: {detected_text}")
        print("-" * 30)
        
        results.append({'Image Name': image_file, 'Detected Plate': detected_text})

        # Show images (optional - might block if not careful, good to comment out for automation)
        # cv2.imshow("Original", original_img)
        # cv2.imshow("Plate", plate_img)
        # cv2.waitKey(0)

    # Save results to Excel
    if results:
        df = pd.DataFrame(results)
        output_file = 'detected_plates.xlsx'
        df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No results to save.")

    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
