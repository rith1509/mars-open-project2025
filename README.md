# 🎙️ Speech Emotion Recognition (SER) Pipeline
## 📘 Project Overview

An end-to-end Speech Emotion Recognition (SER) system that:

Processes audio files into log-mel spectrograms (2-second middle crop at 48 kHz).

Trains a custom Inception-style CNN from scratch for emotion classification.

Evaluates model performance with accuracy, F1 score, and per-class metrics.

Provides local inference through a Streamlit web app and a test script.

## 🚀 Repository Structure
speech_emotion/
├── app.py                    # Streamlit web app for local inference
├── inception.ipynb           # Jupyter/Kaggle notebook for training & evaluation
├── inception_ser_final.pt    # Trained model weights (PyTorch state_dict)
├── test_model.py             # Standalone local test script
├── requirements.txt          # Python dependencies
└── README.md                 # This file

## ✨ Features

Inception-style CNN trained from scratch on log-mel spectrograms.

Data preprocessing: crop/pad audio to middle 2 seconds, compute log-mel features.

Class balancing with weighted sampling and optional focal loss.

Metrics: Confusion matrix, classification report, per-class accuracy.

Inference:

app.py: Streamlit web app to upload .wav files and see predictions.

test_model.py: Command-line test script for quick local inference.

## 🧠 Tech Stack

Language: Python 3.9+

Core Libraries: PyTorch, torchaudio, librosa, numpy, scikit-learn

Visualization: matplotlib, seaborn, tqdm

Web Framework: Streamlit

## 🏋️‍♂️ Training & Evaluation (Kaggle or Local)

Open inception.ipynb in Kaggle or Jupyter.

Load and preprocess your dataset into a pandas DataFrame with file paths and emotion labels.

Train the model using:

AdamW optimizer, CosineAnnealingLR scheduler

Early stopping on weighted F1 score

Evaluate:

✅ Overall accuracy ≥ 80%

✅ Weighted F1 ≥ 80%

✅ Per-class accuracy ≥ 75%

Save the best model weights as inception_ser_final.pt.

## 💻 Local Inference (Streamlit App)

Install dependencies:

pip install -r requirements.txt


Place inception_ser_final.pt in the same directory as app.py.

Run the app:

streamlit run app.py


Open http://localhost:8501
, upload a .wav file, and view:

## 🎧 Audio playback

🧩 Predicted emotion and confidence

📊 Probability bar chart

🧪 Test Script (test_model.py)

A quick local inference script to classify emotions from .wav files using the trained model.

Example Usage
python test_model.py


Example output:

<img width="1274" height="464" alt="image" src="https://github.com/user-attachments/assets/d77e6a15-e52e-4671-a6fb-0942c01d02f8" />

<img width="801" height="402" alt="image" src="https://github.com/user-attachments/assets/388f9990-859a-4efc-a658-0c8fe5cca3d2" />


## 📦 Requirements
torch
torchvision
torchaudio
librosa
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
streamlit


Install with:

pip install -r requirements.txt

## 📈 Performance Targets
Metric	Target
Overall Accuracy	≥ 80%
Weighted F1 Score	≥ 80%
Per-Class Accuracy	≥ 75%
