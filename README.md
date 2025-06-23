# Speech Emotion Recognition (SER) Pipeline

**Project Overview**

This repository implements an end-to-end Speech Emotion Recognition (SER) pipeline, including:
- **Data preprocessing**: load audio files, truncate/pad to middle 2 seconds at 48 kHz, compute log-mel spectrograms.
- **Model training**: a custom Inception-style CNN trained from scratch on spectrogram inputs, with class balancing.
- **Evaluation**: in the training notebook, produce confusion matrix, classification report, and ensure strict criteria:
  - Overall accuracy ≥ 80%
  - Weighted F1 score ≥ 80%
  - Per-class accuracy ≥ 75%
- **Inference & demo**:
  - **Notebook** (`inception.ipynb`): training & evaluation pipeline on Kaggle or local Jupyter.
  - **Model weights** (`inception_ser_final.pt`): saved best model state_dict.
  - **Streamlit app** (`app.py`): upload a `.wav` file locally and get predicted emotion + probabilities.
- **Demo video**: Provided separately via Google Drive link in the submission form (do not include the video file in this repo).

This README is tailored for usage **on Kaggle** for training/experiments and **locally** for running the Streamlit app. It references only the files present in this repository.

---

## 🚀 Repository Structure

speech_emotion/
├── app.py # Streamlit web-app for local inference
├── inception.ipynb # Jupyter/Kaggle notebook: training & evaluation pipeline
├── inception_ser_final.pt # Trained model weights (state_dict) for Inception-style CNN
├── requirements.txt # Python dependencies (for local environment)
└── README.md # This file

markdown
Copy
Edit

- **venv/**: If you set up a local virtual environment; ignored on Kaggle.
- **app.py**: Streamlit application for local inference (upload WAV, display predicted emotion & probabilities).
- **inception.ipynb**: Notebook containing the full training pipeline: data loading, preprocessing, model definition, training loop, evaluation metrics and confusion matrices.
- **inception_ser_final.pt**: Saved PyTorch `state_dict` of the best model from training.
- **requirements.txt**: List of Python packages to install for local usage.
- **README.md**: This documentation.

---

## ✨ Core Features

- **Custom Inception-Style CNN**  
  - Trained from scratch on log-mel spectrogram inputs (middle 2 s at 48 kHz).  
  - No external pretrained backbones.
- **Audio Preprocessing**  
  - Load audio at 48 kHz; pad/truncate to exact middle 2 seconds.  
  - Compute log-mel spectrogram (64 mel bins, hop_length=512), normalize per example.  
  - (Optional augmentations during training in notebook: random 2 s crop, pitch shift.)
- **Class Balancing**  
  - WeightedRandomSampler with inverse-frequency sampling.  
  - Configurable oversample boost factors for underperforming classes.  
  - (Optional) Focal Loss to focus on harder examples.
- **Training Utilities**  
  - PyTorch training loop with AdamW (lr=1e-4) and CosineAnnealingLR.  
  - Early stopping on weighted F1 score (patience configurable).  
  - Logging via tqdm and printed metrics per epoch.
- **Evaluation**  
  - Confusion matrix visualization in the notebook.  
  - Classification report (precision, recall, F1 per class).  
  - Per-class accuracy printed; goal: ≥ 75% per class, ≥ 80% overall accuracy, ≥ 80% weighted F1.
- **Streamlit Web App** (`app.py`)  
  - Upload a `.wav` file locally, extract middle 2 s at 48 kHz, compute log-mel, normalize, feed to trained model.  
  - Display audio playback widget.  
  - Show predicted emotion label + confidence, and bar chart of class probabilities.
- **Reproducibility**  
  - Deterministic train/test splits via `random_state=42` in notebook.  
  - Clear instructions for Kaggle environment and local setup.

---

## 🛠 Tech Stack

- **Python**: 3.7+ on Kaggle; locally 3.9+ recommended  
- **Audio Processing**: `librosa`  
- **Deep Learning**: `torch` / `torchvision` / `torchaudio`  
- **Data Handling & Metrics**: `pandas`, `numpy`, `scikit-learn`  
- **Visualization**: `matplotlib`, `seaborn`, `tqdm`  
- **Web App**: `streamlit` for local deployment  
- **Environment**:  
  - **Kaggle**: built-in GPU/CPU environment; install missing packages via pip in notebook.  
  - **Local**: virtual environment or Conda, install dependencies from `requirements.txt`.

---

## 📂 Dataset & DataFrame Preparation

1. **Dataset Description**  
   - The dataset contains `.wav` audio files labeled with emotion categories (e.g., RAVDESS-like or custom).  
   - Filenames or metadata specify the emotion label for each file.

2. **Building the DataFrame** (in `inception.ipynb`)  
   Prepare lists of file paths and labels, then build a pandas DataFrame:
   ```python
   import os
   import pandas as pd

   # Example: when using Kaggle, audio files are under /kaggle/input/your-dataset/
   audio_dir = "/kaggle/input/your-ser-dataset/"
   paths = []
   labels = []
   for root, _, files in os.walk(audio_dir):
       for fname in files:
           if fname.endswith(".wav"):
               fullpath = os.path.join(root, fname)
               # Parse emotion label from filename, e.g., by your naming convention:
               emotion_label = parse_emotion_from_filename(fname)  # implement this function
               paths.append(fullpath)
               labels.append(emotion_label)

   df = pd.DataFrame({'speech': paths, 'label': labels})
   # Map to 0-based integers if needed:
   unique_labels = sorted(df['label'].unique())
   label_map = {old: new for new, old in enumerate(unique_labels)}
   df['label'] = df['label'].map(label_map)
   print("Label mapping:", label_map)
   print("Label distribution:\n", df['label'].value_counts().sort_index())
Implement parse_emotion_from_filename(...) according to your dataset’s naming scheme.

Ensure labels are integers 0..C-1. Keep the same mapping for training and inference.

Train/Test Split
In the notebook:

python
Copy
Edit
from sklearn.model_selection import train_test_split
paths = df['speech'].tolist()
labels = df['label'].tolist()
X_train, X_test, y_train, y_test = train_test_split(
    paths, labels, test_size=0.2, stratify=labels, random_state=42
)
Use X_train, y_train for training (with augment=True), and X_test, y_test for validation (augment=False).

📝 Preprocessing & Feature Extraction
In inception.ipynb, define:

python
Copy
Edit
import numpy as np
import librosa
import random

def extract_logmel_spectrogram_middle2s(
    file_path,
    sr=48000,
    n_mels=64,
    duration=2.0,
    hop_length=512,
    augment=False
):
    """
    Load audio at sampling rate sr, extract exactly middle duration seconds (pad if shorter or random-crop if augment=True),
    optional pitch-shift augmentation, then compute log-mel spectrogram, normalize per-example.
    Returns: numpy array [1, n_mels, n_frames] (float32).
    """
    y, _ = librosa.load(file_path, sr=sr)
    target_len = int(duration * sr)
    if len(y) < target_len:
        pad_amount = target_len - len(y)
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left
        y = np.pad(y, (pad_left, pad_right))
    else:
        if augment and random.random() > 0.5:
            max_offset = len(y) - target_len
            start = np.random.randint(0, max_offset + 1)
        else:
            start = (len(y) - target_len) // 2
        y = y[start:start + target_len]
    # Optional pitch-shift augmentation during training
    if augment and random.random() > 0.5:
        try:
            y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=np.random.uniform(-1,1))
        except:
            pass
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Normalize per-example
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
    return log_mel.astype(np.float32)[None, :, :]  # shape [1, n_mels, n_frames]
Use augment=True for training DataLoader, augment=False for validation/test.

In Kaggle, ensure to install librosa if not present:

python
Copy
Edit
!pip install librosa tqdm
For local inference in app.py, use the same function (without augmentation) to extract spectrogram from uploaded WAV.

🏗 Model Architecture
Enhanced Inception-Style CNN
Defined in both inception.ipynb (for training) and app.py (for inference). Ensure exactly the same model class in both places.

python
Copy
Edit
import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, pool_proj):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out1x1, kernel_size=1),
            nn.BatchNorm2d(out1x1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red3x3, kernel_size=1),
            nn.BatchNorm2d(red3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(red3x3, out3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out3x3),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red5x5, kernel_size=1),
            nn.BatchNorm2d(red5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(red5x5, out5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(out5x5),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class EnhancedInceptionModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=8):
        super().__init__()
        # Initial conv + pool
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Second conv block
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Inception modules
        self.incep3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)    # out=256
        self.incep3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)  # out=480
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.incep4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)   # out=512
        self.incep4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)  # out=512
        self.incep4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)  # out=512
        self.incep4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)  # out=528
        self.incep4e = InceptionModule(528, 256, 160, 320, 32, 128, 128) # out=832
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.incep5a = InceptionModule(832, 256, 160, 320, 32, 128, 128) # out=832
        self.incep5b = InceptionModule(832, 384, 192, 384, 48, 128, 128) # out=1024

        # Global pooling & classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # [B,1024,1,1]
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # x: [B,1,H,W]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.incep3a(x)
        x = self.incep3b(x)
        x = self.maxpool3(x)
        x = self.incep4a(x)
        x = self.incep4b(x)
        x = self.incep4c(x)
        x = self.incep4d(x)
        x = self.incep4e(x)
        x = self.maxpool4(x)
        x = self.incep5a(x)
        x = self.incep5b(x)
        x = self.avgpool(x)         # [B,1024,1,1]
        x = torch.flatten(x, 1)     # [B,1024]
        x = self.dropout(x)
        logits = self.fc(x)         # [B,num_classes]
        return logits
🔧 Kaggle Environment & Setup
Training and evaluation are intended to run on Kaggle. The Kaggle kernel includes most common packages like PyTorch, NumPy, and Pandas. You may need to manually install missing packages like librosa and tqdm.

✅ Steps to Run on Kaggle:
Upload or Mount Your Audio Dataset

Either upload your dataset as a Kaggle Dataset or access it via an existing path like /kaggle/input/....

Open inception.ipynb Notebook

This notebook contains all necessary code:

Imports

DataFrame construction

Feature extraction

Model architecture

Training loop

Evaluation metrics and plots

Install Any Missing Packages
If needed, install librosa and tqdm with:

python
!pip install librosa tqdm
Enable GPU Accelerator

Go to Notebook Settings → Accelerator → GPU.

Prepare DataFrame

Build a DataFrame with two columns:

speech: path to .wav file

label: integer representing emotion

Perform stratified train/test split.

Training Configuration

DataLoader: Use AudioDataset2D with WeightedRandomSampler

Optimizer:

python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
Scheduler:

python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
Loss Function:

Use CrossEntropyLoss with class weights or

Optionally, use FocalLoss for harder class separation

Batch Size: 64 (adjust if GPU memory is limited)

Early Stopping: Monitor weighted F1 score with patience of ~7 epochs

During Each Epoch

Compute validation predictions

Calculate metrics: weighted F1, overall accuracy, per-class accuracy

Print results

Plot confusion matrix for visual insight

Save Best Model

Save the model’s state dict when the best weighted F1 is achieved:

python
torch.save(model.state_dict(), "/kaggle/working/inception_ser_final.pt")
Evaluation

After training, load the saved weights

Evaluate on validation and custom test sets

Ensure all performance thresholds are met:

✅ Per-class accuracy ≥ 75%

✅ Overall accuracy ≥ 80%

✅ Weighted F1 ≥ 80%

Download Trained Weights

Go to "Output" tab in Kaggle

Download inception_ser_final.pt for use in local inference

🌐 Local Inference via Streamlit (app.py)
Use the Streamlit app to locally classify emotion from uploaded .wav files.

✅ Local Setup Instructions
Set Up Environment

Create a virtual or Conda environment with Python 3.9+

Install dependencies:

bash
pip install torch torchvision torchaudio
pip install streamlit librosa numpy pandas matplotlib seaborn tqdm
Or install from requirements.txt:

bash
pip install -r requirements.txt
Prepare Model File

Place inception_ser_final.pt in the same directory as app.py

Run the Streamlit App

bash
cd path/to/speech_emotion
streamlit run app.py
Using the App

Open browser at http://localhost:8501

Upload a .wav file

The app will:

Load the trained model (only once)

Extract the middle 2 seconds at 48 kHz

Compute log-mel spectrogram, normalize input

Run model inference

Display:

🎧 Audio playback

📊 Predicted emotion & confidence

📈 Bar chart of probabilities for all classes

⚠️ Troubleshooting
Issue	Solution
ModuleNotFoundError: No module named 'torch'	Make sure you're in the correct virtual/conda environment and PyTorch is installed
OpenMP warning (libiomp5md.dll already initialized)	Handled via: os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" in app.py
Model file not found	Check that MODEL_WEIGHTS_PATH in app.py matches the actual location of inception_ser_final.pt
Audio not recognized / Bad format	Only .wav files are supported. Use mono 48kHz, or let librosa auto-resample

📈 Workflow Summary
🧪 Training & Evaluation (Kaggle)
Build a labeled DataFrame from your audio files

Perform train/test split

Train the model using:

InceptionModel architecture

Weighted sampler, augmentation, optional focal loss

Validation and metrics per epoch

Save the best model:

inception_ser_final.pt

🖥️ Inference (Local)
Setup your environment

Place app.py and inception_ser_final.pt in the same folder

Run Streamlit:

bash
streamlit run app.py
Upload a .wav → get emotion prediction with probability chart

📜 requirements.txt
Make sure this file contains:

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
On Kaggle: most are preinstalled

On local machine: use pip install -r requirements.txt or install manually
