'''conversion script'''

import os
import librosa
import soundfile as sf
from pathlib import Path

def convert_to_wav_44k(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in input_dir.glob("*"):
        if not file_path.suffix.lower() in [".mp3", ".wav"]:
            continue  # Skip non-audio files

        try:
            print(f"🔍 Processing: {file_path.name}")
            # Load audio (resamples to 44100 Hz and converts to mono)
            y, sr = librosa.load(file_path, sr=44100, mono=True)

            # Output filename
            out_filename = file_path.stem + ".wav"
            out_path = output_dir / out_filename

            # Avoid re-saving if it’s already .wav and has correct format
            if file_path.suffix.lower() == ".wav":
                original_y, original_sr = sf.read(file_path)
                if original_sr == 44100 and len(original_y.shape) == 1:
                    print(f"✅ Already 44.1kHz mono WAV, skipping: {file_path.name}")
                    continue

            # Save new WAV
            sf.write(out_path, y, 44100)
            print(f"✅ Converted and saved: {out_path.name}")

        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")

# 🧪 Example usage
if __name__ == "__main__":
    input_dir = "/Users/rileykrisch/Downloads/Personal Projects/Spectrogram/Youtube Previews"
    output_dir = "/Users/rileykrisch/Downloads/Personal Projects/Spectrogram/normalized mp3 to wav"
    convert_to_wav_44k(input_dir, output_dir)
