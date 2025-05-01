import os
import random
import numpy as np
import librosa
import soundfile as sf
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from librosa.effects import time_stretch as librosa_time_stretch

# =====================
# Config
# =====================
DATA_DIR = '/Users/rileykrisch/Downloads/Personal Projects/Spectrogram/nsynth_wav_from_tfrecord'
METADATA_CSV = os.path.join(DATA_DIR, 'nsynth_metadata.csv')
SAMPLE_RATE = 16000
N_MFCC = 40
BATCH_SIZE = 32
EPOCHS = 50
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# Audio Effect Functions (optional augmentations)
# =====================
def add_noise(y, noise_level=0.005):
    noise = np.random.randn(len(y))
    return y + noise_level * noise

def add_reverb(y):
    ir = np.random.rand(2000) - 0.5
    return scipy.signal.fftconvolve(y, ir, mode='full')[:len(y)]

def apply_compression(y):
    return librosa.effects.preemphasis(y)

def apply_distortion(y, gain=10):
    y = y * gain
    return np.tanh(y)

# =====================
# Dataset
# =====================
class NSynthMFCCDataset(Dataset):
    def __init__(self, file_paths, labels, sr=SAMPLE_RATE, n_mfcc=N_MFCC, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.augment = augment

    def stretch_audio(self, y, min_rate=0.8, max_rate=1.2):
        rate = random.uniform(min_rate, max_rate)
        if len(y) < 2049:
            return y
        try:
            return librosa_time_stretch(y, rate=rate)
        except Exception as e:
            print(f"⚠️ Time-stretch failed: {e}")
            return y

    def apply_effects(self, y, label):
        if label not in [3, 4]:  # Full augmentation for all except class 3 and 4
            if random.random() < 0.5:
                y = add_noise(y, noise_level=0.01)
            if random.random() < 0.3:
                y = add_reverb(y)
            if random.random() < 0.3:
                y = apply_compression(y)
        else:  # Light augmentation for classes 3 and 4
            if random.random() < 0.2:
                y = add_noise(y, noise_level=0.005)
        return y

    def extract_mfcc(self, y, max_len=MAX_LEN):
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        stacked = np.stack([mfcc, delta, delta2])
        stacked = (stacked - np.mean(stacked)) / (np.std(stacked) + 1e-6)
        if stacked.shape[2] < max_len:
            pad_width = max_len - stacked.shape[2]
            stacked = np.pad(stacked, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        else:
            stacked = stacked[:, :, :max_len]
        return torch.tensor(stacked, dtype=torch.float32)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        y, _ = librosa.load(path, sr=self.sr)
        if self.augment:
            y = self.stretch_audio(y)
            y = self.apply_effects(y, label)
        mfcc_tensor = self.extract_mfcc(y)
        return mfcc_tensor, torch.tensor(label, dtype=torch.long)

# =====================
# Model
# =====================
class MFCC_CNN(nn.Module):
    def __init__(self, num_classes):
        super(MFCC_CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

# =====================
# Data Preparation
# =====================
meta = pd.read_csv(METADATA_CSV)
file_paths = [os.path.join(DATA_DIR, fname) for fname in meta['filename']]
labels = list(meta['instrument_family'])
unique_families = sorted(meta['instrument_family'].unique())
family_to_idx = {fam: i for i, fam in enumerate(unique_families)}
labels = [family_to_idx[fam] for fam in meta['instrument_family']]

train_files, val_files, train_labels, val_labels = train_test_split(
    file_paths, labels, test_size=0.1, stratify=labels, random_state=42)

train_dataset = NSynthMFCCDataset(train_files, train_labels, augment=True)
val_dataset = NSynthMFCCDataset(val_files, val_labels, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================
# Training + Evaluation
# =====================
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    print(classification_report(y_true, y_pred))
    return np.mean(np.array(y_true) == np.array(y_pred))

# =====================
# Run + Plotting
# =====================
model = MFCC_CNN(num_classes=len(family_to_idx)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_losses = []
val_accuracies = []

for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_acc = evaluate(model, val_loader)
    train_losses.append(train_loss)
    val_accuracies.append(val_acc)
    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")

# Plotting
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy', color='orange', marker='o')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(model, loader, class_names):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)

            _, predicted = torch.max(outputs, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

# Call it after training
plot_confusion_matrix(model, val_loader, class_names=list(family_to_idx.keys()))
