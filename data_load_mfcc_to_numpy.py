# -*- coding: utf-8 -*-
"""Data Load_MFCC_to Numpy

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1AgHab0cqNCb3ovbdMNE68FwwB44GHR1m
"""

import os
import random
import librosa
import numpy as np
import scipy.signal
from tqdm import tqdm
from google.colab import drive
import deeplake

# === 1. MOUNT GOOGLE DRIVE ===
drive.mount('/content/drive', force_remount=True)
target_dir = "/content/drive/MyDrive/nsynth_subset"
os.makedirs(target_dir, exist_ok=True)

# === 2. LOAD DEEPLAKE DATASETS ===
print("📥 Loading DeepLake datasets...")
dstr = deeplake.open("hub://activeloop/nsynth-train")
dsva = deeplake.open("hub://activeloop/nsynth-val")
dste = deeplake.open("hub://activeloop/nsynth-test")

# === 3. COMBINE DATASETS AND GROUP BY INSTRUMENT FAMILY ===
samples_per_class = 150
instrument_families = {}

# Combine all sets into one list of (dataset, index)
all_sets = [(dstr, i) for i in range(len(dstr))] + \
           [(dsva, i) for i in range(len(dsva))] + \
           [(dste, i) for i in range(len(dste))]

print("📦 Indexing instrument families across all splits...")
for dataset, i in tqdm(all_sets, desc="Indexing"):
    try:
        info = dataset[i].numpy(as_dict=True)
        key = info['id'].decode('utf-8')
        family = info['instrument_family_str'].decode('utf-8')
        if family not in instrument_families:
            instrument_families[family] = []
        instrument_families[family].append((dataset, i))
    except Exception as e:
        print(f"⚠️ Skipped sample at index {i}: {e}")

# === 4. SAMPLE 150 PER FAMILY ===
selected_samples = []
for family, samples in instrument_families.items():
    selected = random.sample(samples, min(samples_per_class, len(samples)))
    selected_samples.extend(selected)

print(f"\n🎯 Total selected samples: {len(selected_samples)} across {len(instrument_families)} families.")

# === 5. AUDIO PIPELINE ===
def bandpass_filter(y, sr, lowcut, highcut):
    b, a = scipy.signal.butter(4, [lowcut / (sr / 2), highcut / (sr / 2)], btype='band')
    return scipy.signal.lfilter(b, a, y)

def combine_band_filters(y, sr, bands_dict):
    combined = np.zeros_like(y)
    for _, (low, high) in bands_dict.items():
        y_band = bandpass_filter(y, sr, low, high)
        combined += y_band
    return combined

def run_pipeline(y, sr, use_filters=False):
    y_sliced = y[:sr * 2]
    if use_filters:
        bands = {"Low": (300, 800), "Mid": (800, 2000), "High": (2000, 3400)}
        return combine_band_filters(y_sliced, sr, bands)
    return y_sliced

# === 6. PROCESS AUDIO AND SAVE MEL SPECTROGRAMS ===
save_dir = os.path.join(target_dir, "spectrogram_arrays")
os.makedirs(save_dir, exist_ok=True)

failed = []

for dataset, idx in tqdm(selected_samples, desc="Processing samples"):
    try:
        item = dataset[idx].numpy(as_dict=True)
        key = item['id'].decode('utf-8')
        audio = item['audio'].astype(np.float32)

        y_proc = run_pipeline(audio, sr=16000, use_filters=False)

        # Create mel spectrogram
        mel = librosa.feature.melspectrogram(y=y_proc, sr=16000, n_fft=2048, hop_length=512, n_mels=128, fmax=16000)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Save to Drive
        np.save(os.path.join(save_dir, f"{key}.npy"), mel_db)

    except Exception as e:
        failed.append((dataset, idx))
        print(f"❌ Failed {key}: {e}")

print(f"\n✅ Done! Saved {len(selected_samples) - len(failed)} spectrograms to {save_dir}")
if failed:
    print(f"⚠️ {len(failed)} samples failed to process.")

# Upgrade DeepLake to make sure you're on v4+
!pip uninstall -y deeplake
!pip install "deeplake<4"

