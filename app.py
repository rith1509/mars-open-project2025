# app.py

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import librosa
import tempfile
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd

# -------------------------------
# 1. Configuration / Globals
# -------------------------------

# Path to your saved model weights (state_dict)
MODEL_WEIGHTS_PATH = "inception_ser_final.pt"  # ADJUST if your file name differs

# Device for inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Audio preprocessing parameters (must match what you used at training)
SR = 48000            # sampling rate used during training
DURATION = 2.0        # seconds of audio used (middle 2s)
N_MELS = 64           # number of mel bins
HOP_LENGTH = 512      # hop length for mel-spectrogram

# Label names: MUST match the order/index mapping used during training.
# E.g., if you had 8 classes in RAVDESS in order [neutral, calm, happy, sad, angry, fearful, disgust, surprised]:
EMOTION_LABELS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]
NUM_CLASSES = len(EMOTION_LABELS)

# -------------------------------
# 2. Model Definitions
#    Must match your training architecture exactly
# -------------------------------

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
    def __init__(self, in_channels=1, num_classes=NUM_CLASSES):
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
        self.incep3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)    # output channels = 64+128+32+32 = 256
        self.incep3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)  # output = 128+192+96+64 = 480
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
        x = self.avgpool(x)    # [B,1024,1,1]
        x = torch.flatten(x, 1)  # [B,1024]
        x = self.dropout(x)
        logits = self.fc(x)     # [B,num_classes]
        return logits

# -------------------------------
# 3. Audio Preprocessing for inference
# -------------------------------

def extract_logmel_spectrogram_middle2s(file_path, sr=SR, n_mels=N_MELS, duration=DURATION, hop_length=HOP_LENGTH):
    """
    Load audio at sampling rate sr, extract exactly middle duration seconds (pad if shorter).
    Compute log-mel spectrogram, normalize to zero mean/unit variance.
    Returns numpy array [1, n_mels, n_frames].
    """
    # 1. Load full audio
    y, _ = librosa.load(file_path, sr=sr)
    target_len = int(duration * sr)

    if len(y) < target_len:
        pad_amount = target_len - len(y)
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left
        y = np.pad(y, (pad_left, pad_right))
    else:
        # take middle segment
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]

    # Compute mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel, ref=np.max)  # [n_mels, n_frames]
    # Normalize per-example
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
    # Return shape [1, n_mels, n_frames]
    return log_mel.astype(np.float32)[None, :, :]

# -------------------------------
# 4. Load Model (cached)
# -------------------------------

@st.cache_resource(show_spinner=False)
def load_model(weights_path: str = MODEL_WEIGHTS_PATH):
    """
    Load the EnhancedInceptionModel and its state_dict from weights_path.
    """
    model = EnhancedInceptionModel(in_channels=1, num_classes=NUM_CLASSES)
    # Load state_dict
    map_location = DEVICE
    try:
        state = torch.load(weights_path, map_location=map_location)
        model.load_state_dict(state)
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        raise
    model.to(DEVICE)
    model.eval()
    return model

# -------------------------------
# 5. Streamlit App UI
# -------------------------------

st.set_page_config(page_title="Audio Emotion Recognition", layout="centered")
st.title("🎤 Audio Emotion Recognition")

st.markdown(
    """
    Upload a `.wav` file (voice recording). The app extracts the middle 2 seconds at 48 kHz,
    computes a log-mel spectrogram, and runs it through the pretrained Inception-style SER model
    to predict the emotion.
    """
)

# Load the model once
with st.spinner("Loading model..."):
    model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])
if uploaded_file is not None:
    # Save to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        tfile.write(uploaded_file.read())
        tfile.flush()
        filepath = tfile.name

        # Show audio playback
        st.audio(filepath, format="audio/wav")

        # Preprocess
        with st.spinner("Preprocessing audio..."):
            spec = extract_logmel_spectrogram_middle2s(filepath, sr=SR, n_mels=N_MELS, duration=DURATION, hop_length=HOP_LENGTH)
            # Convert to tensor [1,1,H,W]
            tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # Inference
        with st.spinner("Running inference..."):
            with torch.no_grad():
                logits = model(tensor)  # [1, num_classes]
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                pred_conf = float(probs[pred_idx])
                pred_label = EMOTION_LABELS[pred_idx]

        # Display result
        st.markdown(f"**Predicted Emotion:** `{pred_label}`  \nConfidence: {pred_conf*100:.1f}%")

        # Bar chart of probabilities
        df_probs = pd.DataFrame({
            "emotion": EMOTION_LABELS,
            "probability": probs
        }).sort_values("probability", ascending=False)
        st.subheader("Class probabilities:")
        st.bar_chart(data=df_probs.set_index("emotion"))

    finally:
        # Clean up temporary file
        try:
            tfile.close()
            os.remove(filepath)
        except Exception:
            pass

st.markdown("---")
st.markdown(
    "- The model expects 48 kHz audio; shorter files are zero-padded to 2s, longer truncated to middle 2s.  \n"
    "- Ensure `MODEL_WEIGHTS_PATH` and `EMOTION_LABELS` match your training setup.  \n"
    "- If certain classes underperform, consider retraining with more augmentation or test-time augmentation.  \n"
)

