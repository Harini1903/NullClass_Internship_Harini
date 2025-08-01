🎯 Task 1: Activation Map Visualization – Emotion Detection

Task_1_Activation_Maps/
│
├── model/                        
├── model_weights.weights.h5     
├── images/                      
├── visualizations/              
├── emotional_detection.ipynb    
├── requirements.txt             
└── README.md 

📌 Objective
To visualize CNN layer activations using sample face images. These visualizations help interpret how the model perceives emotions and what spatial features it learns during training.

✅ Features
Loads a pre-trained model architecture and weights
Preprocesses grayscale face images (48x48 resolution)
Visualizes activations from a specified convolutional layer
Saves the activation maps as .png files
Clean, organized output in the visualizations/ folder

🧠 Technologies Used
Python
TensorFlow / Keras
OpenCV
NumPy
Matplotlib
Jupyter Notebook

🧪 How to Run
Install dependencies:
pip install -r requirements.txt
Run the Notebook: Open emotional_detection.ipynb and execute all the cells.

Test Activation: Make sure you place test images in the images/ folder and call:

visualize_activation("images/happy_test.jpg")
Check Outputs: Saved activation maps will appear in the visualizations/ folder.

📷 Sample Output
activation_map_example

📊 Results
Activation maps show how filters in early CNN layers respond to various facial regions.
Helps debug and improve feature learning in emotion recognition models.

📁 Requirements
See requirements.txt. Main libraries:

tensorflow
keras
opencv-python
matplotlib
numpy


