import os
import numpy as np
import librosa
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from mpl_toolkits.mplot3d import Axes3D

# === MUST BE DEFINED FIRST ===
audio_files = [
    '/Users/rileykrisch/Downloads/Personal Projects/Spectrogram/Split Stems Demucs/_Users_rileykrisch_Downloads_Youtube_download_script.py/drums.wav'
]
all_bands = {
    'Sub Bass': (20, 80),
    'Bass': (80, 250),
    'Low Mids': (250, 500),
    'Mids': (500, 2000),
    'High Mids': (2000, 6000),
    'Air': (6000, 16000)
}

def bandpass_filter(y, sr, lowcut, highcut, order=4):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=sr, output='sos')
    return sosfilt(sos, y)

def plot_mel_3d_spectrogram(y, sr, title="3D Mel Spectrogram", elev=30, azim=120):
    n_fft = 2048
    hop_length = 512
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128, fmax=16000)
    D = librosa.power_to_db(S, ref=np.max)
    mel_frequencies = librosa.mel_frequencies(n_mels=128, fmax=16000)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=hop_length)
    T, F = np.meshgrid(times, mel_frequencies)

    # === 3D Plot ===
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, F, D, cmap='plasma', linewidth=0, antialiased=True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mel Frequency (Hz)')
    ax.set_zlabel('Amplitude (dB)')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    # === Final display block ===
    plt.tight_layout()
    plt.show(block=False)
    input("✅ Press Enter to close the plot...")
    plt.close()

def combine_band_filters(y, sr, bands_dict):
    combined = np.zeros_like(y)
    for band_name, (low, high) in bands_dict.items():
        print(f"  ➤ Filtering {band_name} ({low}-{high} Hz)")
        y_band = bandpass_filter(y, sr, low, high)
        combined += y_band
    return combined

def run_pipeline(file_index=0, selected_band_names=None, start_time=0, end_time=60):
    print(f"\n🧪 DEBUG: file_index = {file_index}")
    print(f"🧪 DEBUG: audio_files length = {len(audio_files)}")
    print(f"🧪 DEBUG: audio_files = {audio_files}")

    if file_index >= len(audio_files):
        raise IndexError("File index out of range.")

    audio_path = audio_files[file_index]
    y, sr = librosa.load(audio_path, sr=None)
    print(f"\n🎵 Loaded: {os.path.basename(audio_path)}")
    print(f"   Duration: {len(y)/sr:.2f}s | Sample Rate: {sr}")

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    y_segment = y[start_sample:end_sample]
    print(f"📏 Segment length: {len(y_segment)} samples ({start_time}s → {end_time}s)")

    if selected_band_names is None:
        selected_band_names = list(all_bands.keys())

    selected_bands = {name: all_bands[name] for name in selected_band_names if name in all_bands}
    print(f"🎛️  Applying band filters: {', '.join(selected_bands.keys())}")
    y_combined = combine_band_filters(y_segment, sr, selected_bands)

    plot_title = f"{os.path.basename(audio_path)} - Combined Bands"
    plot_mel_3d_spectrogram(y_combined, sr, title=plot_title)

# === CALL AT BOTTOM (after definitions) ===
run_pipeline(
    file_index=0,
    selected_band_names=None,
    start_time=0,
    end_time=60
)
