# DATA LOADER 

import tensorflow as tf
import numpy as np
import soundfile as sf
from pathlib import Path
import csv
from collections import defaultdict

# CONFIG
TFRECORD_PATH = "/Users/rileykrisch/Downloads/Personal Projects/Spectrogram/nsynth-train.tfrecord"
OUTPUT_WAV_DIR = Path("/Users/rileykrisch/Downloads/Personal Projects/Spectrogram/nsynth_wav_from_tfrecord")
METADATA_CSV = OUTPUT_WAV_DIR / "nsynth_metadata.csv"
SAMPLE_RATE = 16000
SAMPLES_PER_CLASS = 100  # Save 100 per instrument family
MAX_CLASSES = 10         # NSynth has 10 instrument families

OUTPUT_WAV_DIR.mkdir(parents=True, exist_ok=True)

# TFRecord schema
feature_description = {
    'audio': tf.io.FixedLenFeature([64000], tf.float32),  # Already float32
    'pitch': tf.io.FixedLenFeature([], tf.int64),
    'instrument_family': tf.io.FixedLenFeature([], tf.int64),
    'instrument_source': tf.io.FixedLenFeature([], tf.int64),
    'instrument': tf.io.FixedLenFeature([], tf.int64),
    'velocity': tf.io.FixedLenFeature([], tf.int64),
    'note': tf.io.FixedLenFeature([], tf.int64),
    'qualities': tf.io.FixedLenFeature([10], tf.int64),
}

def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

# Load TFRecord
raw_dataset = tf.data.TFRecordDataset(TFRECORD_PATH)
parsed_dataset = raw_dataset.map(_parse_function)

# Class counters
class_counts = defaultdict(int)
records = []

# Process and save samples per class
for example in parsed_dataset:
    family = example['instrument_family'].numpy()

    if class_counts[family] >= SAMPLES_PER_CLASS:
        continue

    audio = example['audio'].numpy()
    audio = np.clip(audio, -1.0, 1.0)

    source = example['instrument_source'].numpy()
    instrument = example['instrument'].numpy()
    pitch = example['pitch'].numpy()
    velocity = example['velocity'].numpy()
    note = example['note'].numpy()
    qualities = example['qualities'].numpy()

    sample_index = class_counts[family]
    filename = f"nsynth_fam{family}_{sample_index:03d}_inst{instrument}_src{source}_pitch{pitch}.wav"
    wav_path = OUTPUT_WAV_DIR / filename
    sf.write(wav_path, audio, SAMPLE_RATE)

    records.append({
        "filename": filename,
        "instrument_family": family,
        "instrument_source": source,
        "instrument": instrument,
        "pitch": pitch,
        "velocity": velocity,
        "note": note,
        "qualities": " ".join(map(str, qualities)),
    })

    class_counts[family] += 1

    if len(class_counts) == MAX_CLASSES and all(c >= SAMPLES_PER_CLASS for c in class_counts.values()):
        break

# Save metadata CSV
with open(METADATA_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print(f"✅ Saved {len(records)} WAV files (100 per instrument family)")
print(f"📄 Metadata saved to: {METADATA_CSV}")
