import os
import yt_dlp
import subprocess
from pathlib import Path

def download_audio(query_or_url, download_dir="Youtube Previews", start_time=0, duration=60):
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    filename = query_or_url.replace(" ", "_").replace("/", "_").replace(":", "") + ".mp3"
    output_path = os.path.join(download_dir, filename)

    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'quiet': True,
        'default_search': 'ytsearch1',  # Will auto-handle search terms or full URLs
        'outtmpl': os.path.join(download_dir, 'temp_audio.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    print(f"🔍 Searching/Downloading: {query_or_url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([query_or_url])
        except Exception as e:
            print(f"❌ Error downloading: {query_or_url} — {e}")
            return

    # Trim using ffmpeg
    print(f"✂️ Trimming {duration}s from {start_time}s...")
    temp_audio_path = os.path.join(download_dir, 'temp_audio.mp3')
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_time), "-i", temp_audio_path,
        "-t", str(duration), "-acodec", "copy", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(temp_audio_path)

    print(f"✅ Saved to: {output_path}")

def run_batch_from_file(input_file, download_dir):
    with open(input_file, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]
    for query in queries:
        download_audio(query, download_dir)

def run_manual_entry(download_dir):
    print("🎧 Enter song titles or YouTube URLs (comma separated):")
    entries = input("▶️  ").split(",")
    for entry in entries:
        download_audio(entry.strip(), download_dir)

# 🧪 MAIN
if __name__ == "__main__":
    print("\n🎼 YouTube Audio Downloader")
    print("1. Batch mode from .txt file")
    print("2. Manual entry")
    mode = input("Choose mode (1 or 2): ")

    # 📁 Set your download directory
    download_dir = "/Users/rileykrisch/Downloads/Personal Projects/Spectrogram/Youtube Previews"

    if mode.strip() == "1":
        input_file = input("📄 Enter path to input .txt file: ").strip()
        run_batch_from_file(input_file, download_dir)
    elif mode.strip() == "2":
        run_manual_entry(download_dir)
    else:
        print("❌ Invalid selection. Exiting.")


download_audio(query_or_url='', 
                    download_dir="youtube_previews", 
                    start_time=0, 
                    duration=60)
'''To run code, execute python file, then choose option 1 for manual entry and single link, OR option 2 for batch download via .txt file'''
