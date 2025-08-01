# AgeEmotion_MovieTheatre
# 🎭 Age & Emotion Detection – Movie Theatre Access Control 🎥

A deep learning project that performs real-time age and emotion detection using webcam feed. Based on detection, it logs data and determines theatre entry eligibility:
- ❌ People aged **<13 or >60** are **not allowed** (marked in red)
- ✅ Others are allowed (marked in green)
- 😐 Emotion is also detected and logged

## 📦 Features

- Real-time webcam-based **age & emotion detection**
- Restricts entry based on **age rules**
- Saves data (timestamp, age, emotion, status) to `output/detection_log.csv`
- 📊 Optional **Tkinter GUI interface**
- Built using **OpenCV**, **TensorFlow/Keras**, **Pandas**, **Tkinter**

---

## 📁 Project Structure

```

AgeEmotion\_MovieTheatre/
│
├── model/                      # Contains trained age & emotion model files
│   ├── age\_model.json
│   ├── age\_model.h5
│   ├── emotion\_model.json
│   └── emotion\_model.h5
│
├── output/
│   └── detection\_log.csv       # Log file (created automatically)
│
├── haarcascade\_frontalface\_default.xml
├── main.py                     # Core application (camera + detection + logging)
├── gui\_app.py                  # GUI version using Tkinter
├── requirements.txt
└── README.md

````

---

## ⚙️ Requirements

- Python 3.8+
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Tkinter (comes with Python)

Install all dependencies:

```bash
pip install -r requirements.txt
````

---

## 🚀 How to Run

### ▶️ Terminal Version

```bash
python main.py
```

### 🖼️ GUI Version

```bash
python gui_app.py
```

---

## 📋 Detection Rules

* If **age < 13** or **age > 60** → ❌ **Not Allowed** (red box)
* Else → ✅ **Allowed** (green box)
* Detected **emotion** is logged alongside

---

## 🧾 CSV Log Output

CSV is saved in: `output/detection_log.csv`

| Timestamp           | Age | Emotion | Status      |
| ------------------- | --- | ------- | ----------- |
| 2025-07-20 15:30:11 | 25  | Happy   | Allowed     |
| 2025-07-20 15:30:22 | 9   | -       | Not Allowed |

---

## 📈 Model Accuracy

* Emotion Model: \~70% (based on FER2013 dataset)
* Age Model: \~68–72% (based on UTKFace pre-trained model)

---

## 🛠️ Future Improvements

* Add gender detection
* Improve age range granularity
* Add alert sounds for restricted entry
* Export logs as PDF or Excel

---

## 🧑‍💻 Author

**Harini P**
Internship Project @ NullClass
July 2025

---
