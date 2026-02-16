
import cv2
import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

def train_model():
    dataset_path = "dataset"
    embedding_model_path = "openface_nn4.small2.v1.t7"
    detector_proto = "model/deploy.prototxt"
    detector_model = "model/res10_300x300_ssd_iter_140000.caffemodel"

    # Load Face Detector
    print("Loading Face Detector...")
    detector = cv2.dnn.readNetFromCaffe(detector_proto, detector_model)

    # Load Embedding Model
    print("Loading Embedding Model...")
    embedder = cv2.dnn.readNetFromTorch(embedding_model_path)

    image_paths = []
    labels = []
    
    # Traverse dataset directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                path = os.path.join(root, file)
                label = os.path.basename(root) # Folder name is the label
                image_paths.append(path)
                labels.append(label)

    if not image_paths:
        print("No images found in dataset directory!")
        return

    known_embeddings = []
    known_names = []

    print(f"Processing {len(image_paths)} images...")

    for (i, path) in enumerate(image_paths):
        print(f"[INFO] Processing image {i+1}/{len(image_paths)}: {path}")
        name = labels[i]

        image = cv2.imread(path)
        if image is None:
            continue
            
        (h, w) = image.shape[:2]
        
        # Construct blob for face detection
        image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), 
                                           (104.0, 177.0, 123.0), swapRB=False, crop=False)
        
        detector.setInput(image_blob)
        detections = detector.forward()

        # Find the face with the highest confidence
        # Find the face with the highest confidence
        detected_face = None
        if len(detections) > 0:
            i_max = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i_max, 2]

            if confidence > 0.5:
                box = detections[0, 0, i_max, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure ROI is within bounds
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW >= 20 and fH >= 20: 
                    detected_face = face
                else:
                    print(f"[DEBUG] Detected face too small in {path}, using full image.")
            else:
                print(f"[DEBUG] Low confidence ({confidence}) in {path}, using full image.")
        
        # Fallback: Use the entire image if no face detected (since it's already a crop)
        if detected_face is None:
             detected_face = image

        # Construct blob for embedding
        face_blob = cv2.dnn.blobFromImage(detected_face, 1.0 / 255, (96, 96), 
                                          (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(face_blob)
        vec = embedder.forward()

        known_names.append(name)
        known_embeddings.append(vec.flatten())

    if not known_embeddings:
         print("No faces found/processed. Please check dataset.")
         return

    print("[INFO] Encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(known_names)

    if len(le.classes_) < 2:
        print("[ERROR] The model requires at least 2 different people to distinguish between them.")
        print(f"[INFO] Currently found only 1 person: {le.classes_}")
        print("[ACTION REQUIRED] Please run 'collect_data.py' again to add a second person.")
        return

    print("[INFO] Training model (KNN)...")
    # Using KNN instead of SVM for better handling of small/noisy datasets
    recognizer = KNeighborsClassifier(n_neighbors=3, weights='distance')
    recognizer.fit(known_embeddings, labels)

    # Save the model and label encoder
    print("[INFO] Saving model...")
    with open("svm_model.pkl", "wb") as f:
        f.write(pickle.dumps(recognizer))

    with open("le.pkl", "wb") as f:
        f.write(pickle.dumps(le))

    print("Model training completed successfully.")

if __name__ == "__main__":
    train_model()
