import os
from pathlib import Path
from pydub import AudioSegment

def convert_youtube_previews_to_wav(mp3_dir, wav_output_dir):
    mp3_dir = Path(mp3_dir)
    wav_output_dir = Path(wav_output_dir)
    wav_output_dir.mkdir(parents=True, exist_ok=True)

    mp3_files = list(mp3_dir.glob("*.mp3"))
    if not mp3_files:
        print(f"❌ No MP3 files found in: {mp3_dir}")
        return

    print(f"🎧 Found {len(mp3_files)} YouTube preview(s) to convert.")

    for mp3_file in mp3_files:
        try:
            print(f"🎼 Converting: {mp3_file.name}")
            audio = AudioSegment.from_mp3(mp3_file)

            out_file = wav_output_dir / f"{mp3_file.stem}.wav"
            audio.export(out_file, format="wav")
            print(f"✅ Saved: {out_file.name}")
        except Exception as e:
            print(f"❌ Failed to convert {mp3_file.name}: {e}")

if __name__ == "__main__":
    mp3_folder = "/Users/rileykrisch/Downloads/Personal Projects/Spectrogram/Youtube Previews/"
    wav_output_folder = "/Users/rileykrisch/Downloads/Personal Projects/Spectrogram/Youtube_WAV_FILES/"

    convert_youtube_previews_to_wav(mp3_folder, wav_output_folder)
