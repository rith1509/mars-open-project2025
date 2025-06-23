# Speech Emotion Recognition (SER) Pipeline

**Project Overview**

**Speech Emotion Recognition (SER) Pipeline** is an end-to-end system for classifying emotions from speech audio. This repository contains code to preprocess audio data, train a custom Inception-style CNN model for SER (no external pretrained weights), evaluate performance against strict criteria (per-class accuracy ≥75%, overall accuracy ≥80%, weighted F1 ≥80%), and deploy a Streamlit web app for real-time emotion prediction from `.wav` files.

We use RAVDESS-like datasets (or your specified dataset) where you have file paths and integer emotion labels in a DataFrame. The pipeline extracts the middle 2 seconds at 48 kHz, computes log-mel spectrograms, and trains an Inception-style CNN from scratch with class-balancing and optional focal loss. A Streamlit app loads the trained weights and serves predictions.

🔗 **Demo Video**: \[Link to your 2-min demo video showing the web-app in action]

---

## 🚀 Repository Structure

```
├── notebooks/
│   └── training_notebook.ipynb    # Jupyter notebook with full training pipeline, experiments, metrics
├── app.py                         # Streamlit web-app for uploading .wav and showing predicted emotion
├── inception_ser_final.pt         # Trained model weights (state_dict) for Inception-style CNN
├── requirements.txt               # List of pip/conda dependencies
└── README.md                      # This file
```

---

## ✨ Features

* **Custom Inception-Style CNN**: Trained from scratch on log-mel spectrogram inputs (middle 2 seconds at 48 kHz), no external pretrained backbones.
* **Audio Preprocessing**:

  * Load at 48 kHz, truncate/pad to middle 2 seconds per file.
  * Compute log-mel spectrogram (64 Mel bins), normalize per example.
  * (Optional augmentations during training: pitch-shift, random cropping of 2s segment for long audio.)
* **Class Balancing**:

  * WeightedRandomSampler with inverse-frequency sampling, with configurable oversample boost for underperforming classes.
  * Option to use Focal Loss to focus on hard examples.
* **Training Utilities**:

  * PyTorch training loop with AdamW (lr=1e-4) and CosineAnnealingLR.
  * Early stopping on weighted F1 score (patience configurable).
  * Logging via tqdm and printed metrics per epoch.
* **Evaluation**:

  * Confusion matrix visualization.
  * Classification report (precision, recall, F1 per class).
  * Per-class accuracy printed; aims: each class ≥75% accuracy, overall accuracy ≥80%, weighted F1 ≥80%.
* **Streamlit Web App**:

  * Upload a `.wav` file; load middle 2s at 48 kHz; compute log-mel; normalize; feed to trained model.
  * Display audio playback, predicted emotion label + confidence, bar chart of class probabilities.
  * Easy setup: conda environment, install dependencies, run `streamlit run app.py`.
* **Extensibility**:

  * Easily adjust oversample\_boost factors for underperforming classes.
  * Toggle focal loss in training.
  * Adjust hyperparameters: batch size, learning rate, epochs.
* **Reproducibility**:

  * Deterministic splits via `random_state=42`.
  * Clear instructions to build DataFrame from your list of file paths and labels.
  * README details environment setup, training steps, evaluation, and deployment.

---

## 🛠 Tech Stack

* **Language**: Python 3.9+
* **Audio Processing**: librosa for loading audio and computing mel-spectrograms.
* **Deep Learning**: PyTorch for model definition, training, and inference.
* **Data Handling & Metrics**:

  * pandas for DataFrame operations.
  * scikit-learn for train/test split, metrics (accuracy, F1, confusion matrix).
* **Visualization**: matplotlib and seaborn for confusion matrix plots.
* **Web App**: Streamlit for interactive UI to upload `.wav` and show predictions.
* **Environment**: Conda environment with CPU/GPU support for PyTorch; example instructions provided.

---

## 📂 Dataset

* Prepare a pandas DataFrame `df` with two columns:

  * `speech`: full file paths to `.wav` audio files.
  * `label`: integer emotion labels (0-based). Map original labels to 0..(C-1) if needed.
* Example RAVDESS: filenames indicate actor and emotion; parse filenames to build `paths` and `labels`.
* **Train/Test Split**: 80% train, 20% test (stratified by label).
* **Custom Validation**: After training, evaluate on a held-out custom test set (provided by judges) in the same pipeline.

---

## 🎯 Evaluation Criteria

* **Primary Judging**: Confusion matrix on validation data & custom test data.
* **Target Metrics**:

  * **Overall accuracy ≥ 80%**
  * **Weighted F1 score ≥ 80%**
  * **Per-class accuracy ≥ 75%** for all emotion classes.
* Inspect confusion matrix to see which classes underperform; adjust class balancing / augmentation.

---

## 📝 Preprocessing & Feature Extraction

In training scripts or notebook, use the following function to extract log-mel spectrogram of the middle 2 seconds:

```python
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
    Load audio, extract exactly middle `duration` seconds at sampling rate `sr`,
    pad if shorter, optionally random-crop if augment=True,
    compute log-mel spectrogram, normalize to zero mean/unit variance.
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
    # Optional pitch-shift augmentation
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
```

For alternative 1D pipeline (CNN+LSTM), extract frame-wise MFCC+delta+log-mel+other spectral features aligned to a fixed frame length.

---

## 🏗 Model Architecture

### Enhanced Inception-Style CNN (from scratch)

* **Input**: `[B, 1, n_mels, n_frames]`, e.g. `1 × 64 × ~188` (for 2s @ 48kHz with hop\_length=512 → \~188 frames).
* **Initial Conv Blocks**:

  * Conv2d(1→64, kernel=7, stride=2, padding=3) → BN → ReLU → MaxPool(3×3, stride=2)
  * Conv2d(64→64, 1×1) → BN → ReLU → Conv2d(64→192, 3×3, padding=1) → BN → ReLU → MaxPool(3×3, stride=2)
* **Inception Modules**:

  * Multiple InceptionModule blocks (as in GoogLeNet) with branches (1×1, 3×3 after reduction, 5×5 after reduction, pooling+proj).
  * Sequence: incep3a, incep3b, pool, incep4a–4e, pool, incep5a–5b.
* **Global Pool & Classifier**:

  * AdaptiveAvgPool2d → Flatten → Dropout(0.4) → Linear(1024 → num\_classes).
* **No Pretrained Weights**: Entire model trained from scratch on your SER data.

### Alternative 1D CNN+BiLSTM+Self-Attention (optional)

* Input: frame-wise features `[B, T, D]`, permuted to `[B, D, T]`.
* CNN 1D stack: D→128→256→512→512, then permute `[B, T, 512]`.
* BiLSTM: input 512, hidden 128 (bidirectional → 256 output).
* Self-Attention: MultiheadAttention(embed\_dim=256, num\_heads=4) on LSTM outputs.
* Mean pooling over time → `[B, 256]` → classifier head `[256→128→num_classes]`.
* Use if 2D approach underperforms; Inception-2D often yields better performance on spectrogram images.

---

## 🛠 Environment & Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/ser-pipeline.git
   cd ser-pipeline
   ```

2. **Conda environment (Windows example)**:

   ```powershell
   conda create -n ser_app_env python=3.9 -y
   conda activate ser_app_env
   conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
   pip install streamlit librosa numpy pandas matplotlib seaborn tqdm scikit-learn
   ```

3. **Verify files**:

   * `scripts/train.py`, `scripts/evaluate.py`, `scripts/inference.py`, `app.py`, `requirements.txt`, `inception_ser_final.pt`.
   * `requirements.txt` example:

     ```
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
     ```
   * Run `pip install -r requirements.txt` if you filled it out.

---

## 📓 Training (training\_notebook.ipynb or train.py)

1. **Prepare DataFrame**:

   ```python
   import pandas as pd
   # Example lists:
   paths = ["/path/to/audio1.wav", "/path/to/audio2.wav", ...]
   labels = ["happy", "sad", "angry", ...]  # or integer labels
   df = pd.DataFrame({'speech': paths, 'label': labels})
   # Map to 0-based ints if needed:
   unique_labels = sorted(df['label'].unique())
   label_map = {old: new for new, old in enumerate(unique_labels)}
   df['label'] = df['label'].map(label_map)
   print("Label mapping:", label_map)
   print("Label distribution:\n", df['label'].value_counts().sort_index())
   ```

2. **Edit hyperparameters**:

   * Batch size: `64`
   * Learning rate: `1e-4`
   * Weight decay: `1e-5`
   * Epochs: e.g. `50` (with early stopping).
   * Oversample boost: e.g. `{1:2.0,2:2.0,3:2.0,5:2.0}` for underperforming classes.
   * Use focal loss: `use_focal=True` if certain classes are hard.

3. **Run training** (example in `scripts/train.py`):

   ```bash
   python scripts/train.py --data_csv metadata.csv --batch_size 64 --lr 1e-4 --epochs 50 --oversample_boost 1:2.0,3:2.0 --use_focal False
   ```

   * Internally: loads DataFrame, calls `run_inception_pipeline(...)`, saves best weights `inception_ser_final.pt`.
   * Prints per-epoch val F1 / Acc, final confusion matrix & per-class accuracies.

4. **Inspect metrics**:

   * Confirm overall accuracy ≥80%, weighted F1 ≥80%.
   * Examine confusion matrix: per-class accuracy ≥75%.
   * If not met: adjust oversample boost, augmentations, model capacity, learning rate, or collect more data.

5. **Save best model**:

   * `train.py` should save state\_dict to `inception_ser_final.pt`.

---

## 📊 Evaluation (evaluate.py)

* Load saved model weights.
* Run on held-out validation or custom test set.
* Print classification report and confusion matrix.
* Example:

  ```bash
  python scripts/evaluate.py --model_weights inception_ser_final.pt --data_csv val_metadata.csv
  ```
* Ensure metrics meet criteria. If not, revisit training hyperparameters or data augmentation.

---

## 🔍 Inference Script (inference.py)

* CLI to predict emotion for one or multiple `.wav` files.
* Example usage:

  ```bash
  python scripts/inference.py --model_weights inception_ser_final.pt --file path/to/audio.wav
  ```
* Internally:

  * Load model architecture + weights.
  * Preprocess audio: extract middle 2s log-mel spectrogram, normalize.
  * Convert to tensor `[1,1,H,W]`, move to DEVICE.
  * Run `model(tensor)`, apply softmax, output predicted label and confidence.
* Extendable: batch inference on folder.

---

## 🌐 Streamlit Web App (app.py)

Provides UI to upload a `.wav` file and get predicted emotion:

1. **Ensure** you are in the project directory containing `app.py` and `inception_ser_final.pt`.
2. **Run**:

   ```bash
   streamlit run app.py
   ```
3. **App behavior**:

   * Upload a WAV file via browser UI.
   * The app loads the model once (cached).
   * Shows audio playback widget.
   * Extracts middle 2s at 48 kHz → log-mel spectrogram → normalize → tensor `[1,1,H,W]`.
   * Runs inference, displays predicted emotion label + confidence.
   * Shows bar chart of full class probability distribution.
4. **Requirements**:

   * `inception_ser_final.pt` matches the architecture in `app.py`.
   * `EMOTION_LABELS` in `app.py` matches the label mapping used during training.
   * If you see “No module named 'torch'” error: activate `ser_app_env` before running `streamlit run app.py`.
   * The environment variable `KMP_DUPLICATE_LIB_OK=TRUE` is set in `app.py` to bypass OpenMP conflicts if any.

---

## ⚙️ Windows-Specific: Running Streamlit

After activating your conda environment (`ser_app_env`) and installing dependencies:

```powershell
cd C:\Users\RITH\OneDrive\Desktop\speech_emotion
streamlit run app.py
```

* If you get `ModuleNotFoundError: No module named 'torch'`, verify:

  * PyTorch is installed in `ser_app_env`: `conda list | findstr torch`.
  * You run `streamlit run app.py` after `conda activate ser_app_env`.
  * In VSCode or other IDE, ensure the integrated terminal uses the correct environment.

---

## 📈 System Workflow

1. **Data Preparation**

   * Collect audio `.wav` file paths and corresponding emotion labels.
   * Build pandas DataFrame `df` with columns `speech` and `label`.
   * Map original labels (e.g., emotion names) to 0-based integers.

2. **Feature Extraction**

   * For training: use `extract_logmel_spectrogram_middle2s` with `augment=True` for training set, `augment=False` for validation.
   * For 1D pipeline: extract frame-wise MFCC+delta+log-mel+other spectral features.

3. **Model Training**

   * Initialize `EnhancedInceptionModel`.
   * Use WeightedRandomSampler to balance classes; optionally oversample underperforming classes via `oversample_boost`.
   * Use AdamW (lr=1e-4, weight\_decay=1e-5) and CosineAnnealingLR over epochs.
   * Loss: CrossEntropyLoss with class weights or FocalLoss.
   * Early stopping on validation weighted F1 (patience \~7 epochs).
   * Save best `state_dict` (e.g., `inception_ser_final.pt`).

4. **Model Evaluation**

   * Load best weights, evaluate on held-out test set.
   * Print classification report, confusion matrix.
   * Confirm: per-class accuracy ≥75%, overall accuracy ≥80%, weighted F1 ≥80%.
   * If criteria not met: adjust oversample boost, augmentations, hyperparameters, or gather more data.

5. **Inference & CLI**

   * Provide `inference.py` for batch or single-file predictions.

6. **Web App Deployment**

   * Build `app.py` (Streamlit) as above.
   * Run `streamlit run app.py`, upload `.wav`, view predicted emotion and probability chart.
   * Optionally host on Streamlit Cloud or other service.

7. **Documentation & Demo**

   * Include this README.md.
   * Provide demo video showing installation, training small sample, running web app.

---

## 📂 Deliverables

1. **GitHub Repository** containing:

   * `notebooks/training_notebook.ipynb`: Full code for data loading, feature extraction, model definition, training loops, metrics, plots.
   * `inception_ser_final.pt`: Trained model weights (state\_dict).
   * `scripts/train.py`, `scripts/evaluate.py`, `scripts/inference.py`: CLI scripts for training, evaluation, inference.
   * `app.py`: Streamlit web app for real-time emotion prediction.
   * `requirements.txt`: Dependencies.
   * `README.md`: This file.
   * `demo_video.mp4` (or link): 2-minute demo showing web-app usage.

2. **Trained Model**:

   * Save best weights under `inception_ser_final.pt`. Must match architecture in `app.py`.

3. **Python Scripts**:

   * `inference.py`: Accepts path(s) to `.wav` and prints predicted label + confidence.
   * `evaluate.py`: Load validation DataFrame, run model, print classification report, confusion matrix.

4. **README.md**: This file.

5. **Demo Video**: Show environment setup, run web app, upload audio, see predictions, and sample confusion matrix.

6. **Web App**: Hosted or instructions for local run.

---

## 🔧 Troubleshooting

* **“No module named 'torch'” in Streamlit**: Ensure `streamlit run app.py` is executed in the same conda environment where PyTorch is installed.
* **OpenMP error (libiomp5md.dll)**: In `app.py`, `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` is set at top.
* **Model file location**: Ensure `inception_ser_final.pt` is in same directory or adjust `MODEL_WEIGHTS_PATH`.
* **Audio sampling**: App expects 48kHz. librosa will resample if input differs. Short files are zero-padded, longer truncated to middle 2s.
* **Underperforming classes**: Check confusion matrix; use `oversample_boost` in training, more augmentation, adjust hyperparameters.
* **GPU vs CPU**: If GPU available, training/inference uses it; for CPU-only, reduce batch size if needed.
* **Streamlit caching**: Model loaded once via `@st.cache_resource`. Restart app if weights or code change.

---

## 📖 Example Usage

1. **Prepare DataFrame** in notebook:

   ```python
   import pandas as pd
   paths = ["/path/to/audio1.wav", "/path/to/audio2.wav", ...]
   labels = ["happy", "sad", "angry", ...]
   df = pd.DataFrame({'speech': paths, 'label': labels})
   unique_labels = sorted(df['label'].unique())
   label_map = {old: new for new, old in enumerate(unique_labels)}
   df['label'] = df['label'].map(label_map)
   print("Label mapping:", label_map)
   print(df['label'].value_counts().sort_index())
   ```

2. **Train**:

   ```bash
   python scripts/train.py --data_csv data/metadata.csv --batch_size 64 --lr 1e-4 --epochs 50 --oversample_boost 1:2.0,3:2.0
   ```

   * Saves `inception_ser_final.pt`.

3. **Evaluate**:

   ```bash
   python scripts/evaluate.py --model_weights inception_ser_final.pt --data_csv data/val_metadata.csv
   ```

4. **Inference CLI**:

   ```bash
   python scripts/inference.py --model_weights inception_ser_final.pt --file path/to/test.wav
   ```

5. **Run Web App**:

   ```bash
   conda activate ser_app_env
   streamlit run app.py
   ```

   * Upload WAV, see predicted emotion and probabilities.

---

## 🤝 Contributions

Open issues or pull requests to:

* Improve model architecture or hyperparameters.
* Add more augmentations (time-stretch, noise injection).
* Experiment with alternative architectures.
* Add test-time augmentation.
* Dockerize the app.
* Host on Streamlit Cloud, Heroku, AWS.

---


## Acknowledgments

* Inspired by SER research on RAVDESS dataset.
* Inception architecture adapted for spectrograms.
* PyTorch tutorials for CNN/RNN/Attention.
* Streamlit examples for audio apps.

---

*End of README.md*
