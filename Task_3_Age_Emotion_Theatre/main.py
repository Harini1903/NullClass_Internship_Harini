import cv2

# Load Haar cascade classifier
face_cascade = cv2.CascadeClassifier(r"C:\Users\harini p\OneDrive\Desktop\AgeEmotion_MovieTheatre\haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Face Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import os
import csv
from datetime import datetime
from keras.models import model_from_json

# Load emotion detection model
def load_emotion_model():
    with open(r"C:\Users\harini p\OneDrive\Desktop\AgeEmotion_MovieTheatre\model\emotion_model.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(r"C:\Users\harini p\OneDrive\Desktop\AgeEmotion_MovieTheatre\model\emotion_model.weights.h5")
    return model

# Load age detection model
def load_age_model():
    with open(r"C:\Users\harini p\OneDrive\Desktop\AgeEmotion_MovieTheatre\model\age_model.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(r"C:\Users\harini p\OneDrive\Desktop\AgeEmotion_MovieTheatre\model\age_model.weights.h5")
    return model

# Initialize log file
def initialize_log_file(log_path):
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    if not os.path.isfile(log_path):
        with open(log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Age", "Emotion", "Status"])  # header

# Append log entry
def log_detection(log_path, age, emotion, status):
    with open(log_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), int(age), emotion, status])

# Emotion categories
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Paths
face_cascade_path = r"C:\Users\harini p\OneDrive\Desktop\AgeEmotion_MovieTheatre\haarcascade_frontalface_default.xml"
log_file_path = r"C:\Users\harini p\OneDrive\Desktop\AgeEmotion_MovieTheatre\output\detection_log.csv"

# Load models
emotion_model = load_emotion_model()
age_model = load_age_model()
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Initialize log
initialize_log_file(log_file_path)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        try:
            # Emotion prediction
            emotion_face = cv2.resize(face, (48, 48))
            emotion_face = cv2.cvtColor(emotion_face, cv2.COLOR_BGR2GRAY)
            emotion_face = np.reshape(emotion_face, (1, 48, 48, 1)) / 255.0
            emotion_pred = emotion_model.predict(emotion_face)
            predicted_emotion = EMOTIONS_LIST[np.argmax(emotion_pred)]

            # Age prediction
            age_face = cv2.resize(face, (64, 64))
            age_face = np.reshape(age_face, (1, 64, 64, 3)) / 255.0
            predicted_age = age_model.predict(age_face)[0][0]

            # Access decision
            if predicted_age < 13 or predicted_age > 60:
                access_status = "Not Allowed"
                box_color = (0, 0, 255)  # Red
            else:
                access_status = "Allowed"
                box_color = (0, 255, 0)  # Green

            # Draw bounding box and info
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(frame, f"Age: {int(predicted_age)}", (x, y - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {predicted_emotion}", (x, y - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Status: {access_status}", (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            # Log data
            log_detection(log_file_path, predicted_age, predicted_emotion, access_status)

        except Exception as e:
            print("Error processing face:", str(e))
            continue

    cv2.imshow("Age and Emotion Detection - Movie Theatre", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
