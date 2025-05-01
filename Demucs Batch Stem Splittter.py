import torchaudio
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from pathlib import Path
import soundfile as sf
import csv
import os


def load_song_metadata(csv_path):
    metadata = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            base_name = Path(row['filename']).stem
            metadata[base_name] = {
                'song_id': row['song_id'],
                'url': row['url'],
                'original_filename': row['filename']
            }
    return metadata


def split_stems_with_demucs(input_dir, output_dir, metadata_csv, log_file):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_song_metadata(metadata_csv)

    # Load Demucs model
    model = get_model(name="htdemucs").cpu().eval()
    print("🎛️ Loaded Demucs model.")

    # Get list of .wav or .mp3 files
    audio_files = list(input_dir.glob("*.mp3")) + list(input_dir.glob("*.wav"))
        
    if not audio_files:
        print("❌ No audio files found in the input directory.")
        return

    # Prepare log file
    with open(log_file, 'w', newline='') as csvfile:
        fieldnames = ["stem_file", "stem_type", "song_id", "url", "original_filename"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for audio_file in audio_files:
            print(f"\n🎧 Splitting stems for: {audio_file.name}")
            base_name = audio_file.stem
            meta = metadata.get(base_name, {
                'song_id': 'unknown',
                'url': 'unknown',
                'original_filename': audio_file.name
            })

            try:
                wav, sr = torchaudio.load(audio_file)
                if wav.shape[0] == 1:
                    wav = wav.repeat(2, 1)
                if wav.shape[0] > 2:
                    wav = wav[:2]

                stems = apply_model(model, wav.unsqueeze(0), split=True, progress=True)[0]
                stem_names = model.sources

                song_output_dir = output_dir / base_name
                song_output_dir.mkdir(parents=True, exist_ok=True)

                for i, name in enumerate(stem_names):
                    stem_path = song_output_dir / f"{name}.wav"
                    sf.write(stem_path, stems[i].cpu().T.numpy(), sr)
                    print(f"✅ Saved: {stem_path.name}")

                    writer.writerow({
                        'stem_file': str(stem_path),
                        'stem_type': name,
                        'song_id': meta['song_id'],
                        'url': meta['url'],
                        'original_filename': meta['original_filename']
                    })

            except Exception as e:
                print(f"❌ Error processing {audio_file.name}: {e}")

    # ✅ Confirmation
    print(f"\n📄 Combined log file created at: {log_file}")
    print("Each row includes: stem_file, stem_type, song_id, url, original_filename")


# 🧪 Example usage
if __name__ == "__main__":
    input_path = '/Users/rileykrisch/Downloads/Personal Projects/Spectrogram/Youtube_WAV_FILES'
    output_path = "/Users/rileykrisch/Downloads/Personal Projects/Spectrogram/Split Stems Demucs"
    metadata_csv = "/Users/rileykrisch/Downloads/Personal Projects/Spectrogram/url_mapping.csv"
    log_csv = "/Users/rileykrisch/Downloads/Personal Projects/Spectrogram/stem_log.csv"

    split_stems_with_demucs(input_path, output_path, metadata_csv, log_csv)
