import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
import librosa
import random

# ==== Feature Extraction ====
def extract_logmel_spectrogram_middle2s(file_path, sr=48000, n_mels=64, duration=2.0, hop_length=512, augment=False):
    y, _ = librosa.load(file_path, sr=sr)
    target_len = int(duration * sr)
    if len(y) < target_len:
        pad_left = (target_len - len(y)) // 2
        pad_right = target_len - len(y) - pad_left
        y = np.pad(y, (pad_left, pad_right))
    else:
        start = random.randint(0, len(y) - target_len) if augment and random.random() > 0.5 else (len(y) - target_len) // 2
        y = y[start:start + target_len]
    if augment and random.random() > 0.5:
        try:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.uniform(-1, 1))
        except Exception:
            pass
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
    return log_mel.astype(np.float32)[None, :, :]  # Shape: (1, n_mels, time)

# ==== Inception Module ====
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, pool_proj):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out1x1, 1), nn.BatchNorm2d(out1x1), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red3x3, 1), nn.BatchNorm2d(red3x3), nn.ReLU(inplace=True),
            nn.Conv2d(red3x3, out3x3, 3, padding=1), nn.BatchNorm2d(out3x3), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red5x5, 1), nn.BatchNorm2d(red5x5), nn.ReLU(inplace=True),
            nn.Conv2d(red5x5, out5x5, 5, padding=2), nn.BatchNorm2d(out5x5), nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, 1), nn.BatchNorm2d(pool_proj), nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)

# ==== Enhanced Inception Model ====
class EnhancedInceptionModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=8):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.incep3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.incep3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.incep4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.incep4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.incep4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.incep4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.incep4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        self.incep5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.incep5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x)
        x = self.incep3a(x); x = self.incep3b(x); x = self.maxpool3(x)
        x = self.incep4a(x); x = self.incep4b(x); x = self.incep4c(x); x = self.incep4d(x); x = self.incep4e(x); x = self.maxpool4(x)
        x = self.incep5a(x); x = self.incep5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# ==== Run Inference ====
def main():
    wav_path = r"C:\Users\RITH\OneDrive\Desktop\speech_emotion\03-01-06-01-01-01-21.wav"
    model_path = r"C:\Users\RITH\OneDrive\Desktop\speech_emotion\inception_ser_final.pt"
    labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedInceptionModel(in_channels=1, num_classes=len(labels))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    logmel = extract_logmel_spectrogram_middle2s(wav_path, n_mels=64)  # shape: (1, 64, T)
    input_tensor = torch.tensor(logmel).unsqueeze(0).to(device)  # (1, 1, 64, T)

    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()

    print(f"Predicted emotion label: {labels[pred_class]}")

if __name__ == "__main__":
    main()
