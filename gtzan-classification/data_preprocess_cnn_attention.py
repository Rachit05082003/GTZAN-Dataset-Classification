# -*- coding: utf-8 -*-
"""
Data preprocessing pipeline for GTZAN genre classification using CNN + Attention.
Generates log‑Mel spectrogram images (and optional chroma) ready for a 2‑D CNN.

Steps:
1. Load each audio file (30 s, 22 050 Hz).
2. Compute log‑Mel spectrogram (128 mel bins).
3. Optionally compute chromagram and stack as extra channels.
4. Resize/crop to a fixed size (128 × 128).
5. Normalize per‑sample (zero‑mean, unit‑std).
6. Save as NumPy ``.npz`` files (``X`` – images, ``y`` – integer label).

The script also includes a simple ``train_test_split`` and a placeholder for a CNN‑Attention model.
"""

import os
import glob
import numpy as np
import librosa
from skimage.transform import resize
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_ROOT = os.path.expanduser(
    '/Users/narac0503/GIT/GTZAN Dataset Classification/GTZAN-Dataset-Classification/gtzan-classification/data/gtzan/genres_original'  # adjust if needed
)
OUTPUT_DIR = os.path.expanduser(
    '/Users/narac0503/GIT/GTZAN Dataset Classification/GTZAN-Dataset-Classification/gtzan-classification/data/preprocessed'
)
SAMPLE_RATE = 22050
DURATION = 30.0  # seconds
N_MELS = 128
HOP_LENGTH = 512
IMG_SIZE = (128, 128)  # (freq, time) after resizing
USE_CHROMA = True  # set False to generate single‑channel images only

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_audio(path: str, sr: int = SAMPLE_RATE, duration: float = DURATION):
    """Load an audio file, pad/trim to ``duration`` seconds.
    Returns a 1‑D numpy array.
    """
    y, _ = librosa.load(path, sr=sr, duration=duration)
    expected_len = int(sr * duration)
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))
    else:
        y = y[:expected_len]
    return y

def log_mel_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE):
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S

def chromagram(y: np.ndarray, sr: int = SAMPLE_RATE):
    C = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
    # Resize chroma to have the same frequency dimension as mel (128)
    C_resized = resize(C, (N_MELS, C.shape[1]), order=1, mode='constant', anti_aliasing=True)
    return C_resized

def preprocess_file(path: str):
    y = load_audio(path)
    mel = log_mel_spectrogram(y)
    # Resize time axis to IMG_SIZE[1]
    mel_resized = resize(mel, IMG_SIZE, order=1, mode='constant', anti_aliasing=True)
    if USE_CHROMA:
        chroma = chromagram(y)
        chroma_resized = resize(chroma, IMG_SIZE, order=1, mode='constant', anti_aliasing=True)
        img = np.stack([mel_resized, chroma_resized], axis=0)  # (C, H, W)
    else:
        img = mel_resized[np.newaxis, ...]
    # Normalise per‑sample (zero‑mean, unit‑std)
    img = (img - img.mean()) / (img.std() + 1e-6)
    return img.astype(np.float32)

# ---------------------------------------------------------------------------
# Main preprocessing loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # GTZAN folder structure: <genre>/<track>.wav (or .au)
    genres = sorted([d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))])
    label_map = {g: i for i, g in enumerate(genres)}
    print("Found genres:", label_map)

    X, y = [], []
    for genre in genres:
        pattern = os.path.join(DATASET_ROOT, genre, "*.*")
        for file_path in glob.glob(pattern):
            try:
                img = preprocess_file(file_path)
                X.append(img)
                y.append(label_map[genre])
            except Exception as e:
                print(f"[WARN] Failed processing {file_path}: {e}")

    X = np.stack(X)  # shape (N, C, H, W)
    y = np.array(y, dtype=np.int64)
    print(f"Processed {X.shape[0]} files. Shape: {X.shape}")

    # Train/validation split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Save as compressed npz files
    np.savez_compressed(os.path.join(OUTPUT_DIR, "train.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(OUTPUT_DIR, "val.npz"), X=X_val, y=y_val)
    print("Saved train.npz and val.npz to", OUTPUT_DIR)

    # ---------------------------------------------------------------------
    # Placeholder for a CNN + Attention model (e.g., using PyTorch)
    # ---------------------------------------------------------------------
    # The following is a minimal example; replace with your preferred framework.
    #
    # import torch
    # import torch.nn as nn
    # import torch.nn.functional as F
    #
    # class CBAM(nn.Module):
    #     def __init__(self, channels, reduction=16):
    #         super().__init__()
    #         self.avg_pool = nn.AdaptiveAvgPool2d(1)
    #         self.max_pool = nn.AdaptiveMaxPool2d(1)
    #         self.fc = nn.Sequential(
    #             nn.Conv2d(channels, channels // reduction, 1, bias=False),
    #             nn.ReLU(inplace=True),
    #             nn.Conv2d(channels // reduction, channels, 1, bias=False)
    #         )
    #         self.sigmoid = nn.Sigmoid()
    #
    #     def forward(self, x):
    #         avg_out = self.fc(self.avg_pool(x))
    #         max_out = self.fc(self.max_pool(x))
    #         out = self.sigmoid(avg_out + max_out)
    #         return x * out
    #
    # class SimpleCNN(nn.Module):
    #     def __init__(self, in_channels, num_classes):
    #         super().__init__()
    #         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
    #         self.bn1 = nn.BatchNorm2d(32)
    #         self.cbam1 = CBAM(32)
    #         self.pool = nn.MaxPool2d(2)
    #         self.fc = nn.Linear(32 * (IMG_SIZE[0] // 2) * (IMG_SIZE[1] // 2), num_classes)
    #
    #     def forward(self, x):
    #         x = F.relu(self.bn1(self.conv1(x)))
    #         x = self.cbam1(x)
    #         x = self.pool(x)
    #         x = x.view(x.size(0), -1)
    #         return self.fc(x)
    #
    # # Example usage:
    # # model = SimpleCNN(in_channels=2 if USE_CHROMA else 1, num_classes=len(genres))
    # # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # # criterion = nn.CrossEntropyLoss()
    # # ... (training loop) ...
    # ---------------------------------------------------------------------
