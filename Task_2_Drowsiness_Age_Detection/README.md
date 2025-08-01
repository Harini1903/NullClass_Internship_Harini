Drowsiness-and-Age-Detection-AI

Overview This project is a computer vision application that performs real-time drowsiness detection (eye state classification) and age prediction using deep learning. The solution can process images, video files, or webcam input, drawing bounding boxes around detected faces and displaying pop-up notifications with the number and ages of detected drowsy individuals.

Features Real-time face and drowsiness (closed/open eyes) detection using a trained CNN

Age estimation for each detected "drowsy" face using a separate CNN model

Annotated video/image output with labels and colors (red for drowsy, green for awake)

Pop-up summary of current drowsy count and their ages at the end of each run

GUI and command-line interface for practical use

Project Structure text Drowsiness_Detection_AI/ ├── gui/ # Main app and GUI scripts (e.g., gui.py) ├── model/ # Trained Keras models (.h5 files) ├── dataset/ # (Not included) See setup for links ├── notebooks/ # Model training and experiment notebooks ├── requirements.txt ├── README.md ├── LICENSE ├── report/ # Report and result plots (optional) Dataset Sources Drowsiness/Eye State: Drowsiness_Dataset from Kaggle

Age Estimation: UTKFace Dataset from Kaggle

(Due to size, datasets are not included. Download and extract them to the dataset/ folder as described below.)

Setup Instructions

Environment Use Anaconda or python 3.8+ (Windows, macOS, or Linux)
Recommended: Create a virtual environment:

bash conda create -n tf_env python=3.10 conda activate tf_env Install dependencies:

bash pip install -r requirements.txt 2. Dataset Placement Place your drowsiness dataset:

text dataset/Drowsiness_Dataset/closed_eye/ dataset/Drowsiness_Dataset/open_eye/ Place your UTKFace images:

text dataset/age/UTKFace/ 3. Model Training Notebooks for training are provided in the notebooks/ folder. If you wish to (re)train the models, run the Jupyter notebooks:

train_model.ipynb – for drowsiness/eye state model

train_age_model.ipynb – for age prediction model

Or use the pre-trained models:

Place drowsiness_model.h5 and age_model.h5 in the model/ folder.

Running the Application From command prompt, with virtual environment activated:
bash cd gui python gui.py Controls:

The app will open your webcam by default.

Annotated video will be shown live.

Press q to quit and see the summary pop-up.

For image/video input, modify or extend GUI scripts as needed.

Example Output Real-time window with rectangles (red: sleepy, green: alert) and labels.

Pop-up message at end:

text Total drowsy people detected: 2 Ages: 24, 32
