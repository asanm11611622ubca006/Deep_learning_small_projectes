
import cv2
import numpy as np
import pickle
import os
import datetime
import time

def mark_attendance(name):
    filename = "Attendance.csv"
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("Name,RollNo,Time,Date\n")

    with open(filename, 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            
        # Parse Name and RollNo from the label (assumed format Name.RollNo)
        if '.' in name:
            student_name, student_roll = name.split('.')
        else:
            student_name = name
            student_roll = "Unknown"

        # Check if already marked for today (simplification needed mostly, but preventing spam)
        # For this simple script, we just check if the name is in the list, but ideally check date.
        # Let's simple check if the specific user has been logged *in this session* or simply append.
        # To make it robust for daily:
        
        now = datetime.datetime.now()
        dtString = now.strftime('%H:%M:%S')
        dateString = now.strftime('%Y-%m-%d')
        
        # Check if this person is already marked TODAY
        is_marked = False
        for line in myDataList:
            if student_name in line and dateString in line:
                is_marked = True
                break
        
        if not is_marked:
            f.write(f'{student_name},{student_roll},{dtString},{dateString}\n')
            print(f"Attendance marked for {student_name}")

def recognize_attendance():
    # Paths
    embedding_model_path = "openface_nn4.small2.v1.t7"
    detector_proto = "model/deploy.prototxt"
    detector_model = "model/res10_300x300_ssd_iter_140000.caffemodel"
    svm_model_path = "svm_model.pkl"
    le_path = "le.pkl"

    if not os.path.exists(svm_model_path) or not os.path.exists(le_path):
        print("Error: Model files not found. Please train the model first.")
        return

    # Load Face Detector
    print("Loading Face Detector...")
    detector = cv2.dnn.readNetFromCaffe(detector_proto, detector_model)

    # Load Embedding Model
    print("Loading Embedding Model...")
    embedder = cv2.dnn.readNetFromTorch(embedding_model_path)

    # Load SVM and Label Encoder
    print("Loading SVM Model...")
    with open(svm_model_path, "rb") as f:
        recognizer = pickle.loads(f.read())
    with open(le_path, "rb") as f:
        le = pickle.loads(f.read())
    
    print(f"[DEBUG] Classes loaded: {le.classes_}")

    # Initialize Video Stream
    print("Starting Video Stream...")
    cam = cv2.VideoCapture(0)
    
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.resize(frame, (600, 400))
        (h, w) = frame.shape[:2]

        image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                           (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(image_blob)
        detections = detector.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20: 
                    continue

                face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), 
                                                  (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(face_blob)
                vec = embedder.forward()

                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                
                print(f"[DEBUG] Predictions: {dict(zip(le.classes_, preds))}")

                if proba > 0.6: # Confidence threshold
                    text = f"{name}: {proba * 100:.2f}%"
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    
                    # Mark Attendance
                    mark_attendance(name)
                else:
                    # Unknown face
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_attendance()
