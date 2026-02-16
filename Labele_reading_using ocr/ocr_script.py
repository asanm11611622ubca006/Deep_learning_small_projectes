import pytesseract
from PIL import Image
import os
import datetime

# Configure Tesseract path
# IMPORTANT: Update this path if Tesseract is installed in a different location
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def main():
    # Get the current directory where the script is running
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(current_dir, 'output.txt')
    
    print(f"Scanning images in: {current_dir}")
    print(f"Output will be appended to: {output_file_path}")

    # Separator for new runs
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_header = f"\n\n{'='*20}\nOCR Run: {timestamp}\n{'='*20}\n"
    
    try:
        with open(output_file_path, 'a', encoding='utf-8') as f:
            f.write(run_header)
            
            # Counter for processed images
            processed_count = 0
            
            # Loop through all files in the directory
            for filename in os.listdir(current_dir):
                # Check for common image extensions
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_path = os.path.join(current_dir, filename)
                    
                    print(f"Processing: {filename}...")
                    
                    try:
                        # Open the image
                        img = Image.open(image_path)
                        
                        # Perform OCR
                        text = pytesseract.image_to_string(img)
                        
                        # Write to file
                        f.write(f"\n--- File: {filename} ---\n")
                        f.write(text)
                        f.write("\n")
                        
                        processed_count += 1
                        
                    except Exception as e:
                        error_msg = f"Error processing {filename}: {str(e)}\n"
                        print(error_msg.strip())
                        f.write(f"\n--- Error processing {filename} ---\n")
                        f.write(str(e) + "\n")

        print(f"Done. Processed {processed_count} images.")
        print("Text saved to output.txt")

    except Exception as e:
        print(f"Critical error opening output file: {e}")

if __name__ == "__main__":
    main()
