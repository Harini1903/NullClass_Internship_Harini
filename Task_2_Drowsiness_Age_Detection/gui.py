import cv2
import numpy as np
from tensorflow.keras.models import load_model
model = load_model(r"C:\Users\harini p\OneDrive\Desktop\Drowsiness Detection\model\drowsiness_model.h5")


# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Use webcam (change '0' to video file path if testing with a video)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    drowsy_count = 0

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        try:
            resized_face = cv2.resize(face_roi, (64, 64)).reshape(1, 64, 64, 1) / 255.0
            prediction = model.predict(resized_face)
            pred_label = np.argmax(prediction)
            if pred_label == 0:
                color = (0, 0, 255)
                status = "Drowsy"
                drowsy_count += 1
            else:
                color = (0, 255, 0)
                status = "Awake"
        except:
            color = (255,255,0)
            status = "Unknown"
        
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.putText(frame, f"Drowsy Count: {drowsy_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import messagebox

# Load models (set the correct path)
drowsiness_model = load_model(r"C:\Users\harini p\OneDrive\Desktop\Drowsiness Detection\model\drowsiness_model.h5")
age_model = load_model(r"C:\Users\harini p\OneDrive\Desktop\Drowsiness Detection\model\age_model.h5", compile=False)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
all_drowsy_ages = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    frame_drowsy_ages = []

    for (x, y, w, h) in faces:
        face_img_gray = gray[y:y+h, x:x+w]
        face_img_color = frame[y:y+h, x:x+w]

        # Drowsiness prediction (eye state model)
        try:
            resized_face = cv2.resize(face_img_gray, (64, 64)).reshape(1, 64, 64, 1) / 255.0
            pred = drowsiness_model.predict(resized_face)
            is_drowsy = np.argmax(pred) == 0  # Closed = drowsy in your setup
        except:
            is_drowsy = False

        if is_drowsy:
            color = (0, 0, 255)
            # Age prediction (on color face)
            try:
                age_face = cv2.resize(face_img_color, (64, 64))
                if age_face.ndim == 2:
                    age_face = cv2.cvtColor(age_face, cv2.COLOR_GRAY2BGR)
                age_norm = age_face.reshape(1, 64, 64, 3) / 255.0
                pred_age = int(age_model.predict(age_norm)[0][0])
                frame_drowsy_ages.append(pred_age)
            except:
                pred_age = "?"
            label = f"Drowsy ({pred_age})"
        else:
            color = (0, 255, 0)
            label = "Awake"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Live update on video window
    drowsy_count = len(frame_drowsy_ages)
    cv2.putText(frame, f"Sleeping: {drowsy_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (40,40,255), 2)
    all_drowsy_ages.extend(frame_drowsy_ages)
    cv2.imshow("Drowsiness & Age Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# After loop, show pop-up summary
if all_drowsy_ages:
    root = tk.Tk()
    root.withdraw()
    message = f"Total drowsy people detected: {len(all_drowsy_ages)}\nAges: {', '.join(map(str, all_drowsy_ages))}"
    messagebox.showinfo("Summary", message)
    root.destroy()
else:
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Summary", "No drowsy people detected.")
    root.destroy()
