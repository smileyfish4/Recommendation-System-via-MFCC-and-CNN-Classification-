# =====================
# Imports
# =====================
import os
import random
import numpy as np
import librosa
import scipy.signal
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# =====================
# Config
# =====================
VOCAL_STEM_DIR = "/Users/rileykrisch/Downloads/Personal Projects/Spectrogram/Extracted_Vocals"
STEM_LOG = "/Users/rileykrisch/Downloads/Personal Projects/Spectrogram/vocals_stem_log.csv"
SAMPLE_RATE = 16000
N_MELS = 128
HOP_LENGTH = 256
DURATION = 5  # seconds
MAX_LEN = int(DURATION * SAMPLE_RATE / HOP_LENGTH)
BATCH_SIZE = 16
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# Augmentation Functions
# =====================
def add_reverb(y):
    ir = np.random.rand(2000) - 0.5
    return scipy.signal.fftconvolve(y, ir, mode='full')[:len(y)]

def apply_compression(y):
    return librosa.effects.preemphasis(y)

def apply_distortion(y, gain=10):
    y = y * gain
    return np.tanh(y)

def add_noise(y, noise_level=0.005):
    noise = np.random.normal(0, noise_level, size=len(y))
    return y + noise

# =====================
# Dataset
# =====================
class VocalSpectrogramDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment

    def extract_mel_spectrogram(self, y):
        y = y[:DURATION * SAMPLE_RATE]
        if len(y) < DURATION * SAMPLE_RATE:
            y = np.pad(y, (0, DURATION * SAMPLE_RATE - len(y)), mode='constant')
        harmonic, _ = librosa.effects.hpss(y)
        mel = librosa.feature.melspectrogram(y=harmonic, sr=SAMPLE_RATE, n_fft=1024, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if mel_db.shape[1] < MAX_LEN:
            pad = MAX_LEN - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='constant')
        else:
            mel_db = mel_db[:, :MAX_LEN]

        return torch.tensor(mel_db).unsqueeze(0).float()

    def apply_augmentation(self, y):
        if random.random() < 0.4:
            y = add_noise(y)
        if random.random() < 0.3:
            y = add_reverb(y)
        if random.random() < 0.2:
            y = apply_compression(y)
        if random.random() < 0.2:
            y = apply_distortion(y)
        return y

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        y, _ = librosa.load(path, sr=SAMPLE_RATE)
        if self.augment:
            y = self.apply_augmentation(y)
        mel_tensor = self.extract_mel_spectrogram(y)
        return mel_tensor, torch.tensor(label, dtype=torch.long)

# =====================
# Model
# =====================
class VocalCNN(nn.Module):
    def __init__(self, num_classes):
        super(VocalCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, return_embedding=False, return_logits=False):
        x = self.model(x)
        if return_embedding:
            return x
        x = self.dropout(x)
        logits = self.classifier(x)
        if return_logits:
            return logits
        return logits

# =====================
# Utility Functions
# =====================
def load_vocal_paths_and_labels(root_dir):
    file_paths = []
    labels = []

    for f in os.listdir(root_dir):
        if f.endswith(".wav"):
            full_path = os.path.join(root_dir, f)
            file_paths.append(full_path)
            labels.append(0)  # single class: vocals

    print(f"✅ Loaded {len(file_paths)} vocal files.")
    return file_paths, labels, {"vocals": 0}

def extract_embedding(model, filepath):
    y, _ = librosa.load(filepath, sr=SAMPLE_RATE)
    dataset = VocalSpectrogramDataset([], [])
    mel_tensor = dataset.extract_mel_spectrogram(y).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(mel_tensor, return_logits=True).cpu().numpy().flatten()
    return embedding

def build_embedding_db(model, root_dir):
    db = {}
    for f in os.listdir(root_dir):
        if f.endswith(".wav"):
            path = os.path.join(root_dir, f)
            try:
                emb = extract_embedding(model, path)
                song_id = os.path.splitext(f)[0]
                db[song_id] = emb
            except Exception as e:
                print(f"❗Error processing {f}: {e}")
                continue
    return db

def recommend_similar_songs(input_embedding, db_embeddings, stem_log_path, top_k=5):
    df = pd.read_csv(stem_log_path)
    song_ids = list(db_embeddings.keys())
    vectors = np.array([db_embeddings[s] for s in song_ids])
    similarities = cosine_similarity([input_embedding], vectors)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    print("\n🎵 Recommended Songs:")
    for idx in top_indices:
        sid = song_ids[idx]
        link = df[df['song_id'] == sid]['youtube_link'].values
        print(f"{sid} | Similarity: {similarities[idx]:.4f} | Link: {link[0] if len(link) > 0 else 'No link'}")

def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
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
            output = model(X)
            _, predicted = torch.max(output, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    print(classification_report(y_true, y_pred))
    return np.mean(np.array(y_true) == np.array(y_pred)), y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, class_map):
    labels = list(class_map.keys())
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# =====================
# Main Execution
# =====================
def main():
    file_paths, labels, class_map = load_vocal_paths_and_labels(VOCAL_STEM_DIR)

    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.1, random_state=42
    )  # ⬅️ Fixed here: no stratify!

    train_dataset = VocalSpectrogramDataset(train_files, train_labels, augment=True)
    val_dataset = VocalSpectrogramDataset(val_files, val_labels, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = VocalCNN(num_classes=len(class_map)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_accuracies = []

    for epoch in range(EPOCHS):
        loss = train(model, train_loader, criterion, optimizer)
        acc, y_true, y_pred = evaluate(model, val_loader)
        train_losses.append(loss)
        val_accuracies.append(acc)
        print(f"🧠 Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f} - Val Acc: {acc:.4f}")

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy', color='orange', marker='o')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plot_confusion_matrix(y_true, y_pred, class_map)

    # Build Embedding DB and Recommend
    embedding_db = build_embedding_db(model, VOCAL_STEM_DIR)
    sample_embedding = extract_embedding(model, train_files[0])
    recommend_similar_songs(sample_embedding, embedding_db, STEM_LOG)

if __name__ == "__main__":
    main()
