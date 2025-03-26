
import torchaudio
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from pathlib import Path
import soundfile as sf

def split_stems_with_demucs(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Demucs model
    model = get_model(name="htdemucs").cpu().eval()
    print("🎛️ Loaded Demucs model.")

    # Get list of .wav or .mp3 files
    audio_files = list(input_dir.glob("*.mp3")) + list(input_dir.glob("*.wav"))

    if not audio_files:
        print("❌ No audio files found in the input directory.")
        return

    for audio_file in audio_files:
        print(f"\n🎧 Splitting stems for: {audio_file.name}")

        try:
            # Load audio
            wav, sr = torchaudio.load(audio_file)
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)  # mono → stereo
            if wav.shape[0] > 2:
                wav = wav[:2]  # trim to 2 channels if needed

            # Apply Demucs model
            stems = apply_model(model, wav.unsqueeze(0), split=True, progress=True)[0]
            stem_names = model.sources

            # Prepare output path
            song_output_dir = output_dir / audio_file.stem
            song_output_dir.mkdir(parents=True, exist_ok=True)

            # Save each stem
            for i, name in enumerate(stem_names):
                stem_path = song_output_dir / f"{name}.wav"
                sf.write(stem_path, stems[i].cpu().T.numpy(), sr)
                print(f"✅ Saved: {stem_path.name}")

        except Exception as e:
            print(f"❌ Error processing {audio_file.name}: {e}")

# 🧪 Example usage
if __name__ == "__main__":
    input_path = '/Users/rileykrisch/Downloads/Personal Projects/Spectrogram/normalized mp3 to wav'
    output_path = "/Users/rileykrisch/Downloads/Personal Projects/Spectrogram/Split Stems Demucs"
    split_stems_with_demucs(input_path, output_path)


