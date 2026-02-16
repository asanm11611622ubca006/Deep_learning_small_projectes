
import cv2
import os

def collect_data():
    # Load the specific Haar Cascade file available in the project directory
    detector_path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(detector_path):
        print(f"Error: {detector_path} not found.")
        return

    face_detector = cv2.CascadeClassifier(detector_path)
    
    # Get user input for creating the dataset folder
    print("Enter Student's Roll Number (e.g., 101):")
    roll_no = input().strip()
    print("Enter Student's Name (e.g., John):")
    name = input().strip()
    
    folder_name = f"{name}.{roll_no}"
    dataset_path = os.path.join("dataset", folder_name)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"Created directory: {dataset_path}")
    else:
        print(f"Directory {dataset_path} already exists. Appending to it.")

    # Initialize webcam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    count = 0
    print("Starting video stream. Look at the camera...")
    print("Press 'q' to stop early. Attempting to collect 50 samples.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Save the captured face
            count += 1
            # Save file with a unique name
            file_name_path = os.path.join(dataset_path, f"{name}_{roll_no}_{count}.jpg")
            
            # We can save the grayscale or color face depending on training needs.
            # Usually color is fine for OpenFace, but cropping is good practice.
            # Saving the specific face region
            face_img = frame[y:y+h, x:x+w]
            cv2.imwrite(file_name_path, face_img)
            
            cv2.putText(frame, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            print(f"Captured image {count}")

        cv2.imshow('Face Data Collection', frame)

        # Wait for 'q' to stop or stop after 50 images
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Data collection completed.")

if __name__ == "__main__":
    collect_data()
