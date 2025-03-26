import os
import yt_dlp
import subprocess
from pathlib import Path

def download_30sec_audio_from_youtube(query, download_dir="youtube_previews"):
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    # Set up filename
    filename = query.replace(" ", "_") + ".mp3"
    output_path = os.path.join(download_dir, filename)

    # yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'quiet': True,
        'default_search': 'ytsearch1',  # search YouTube and get first result
        'outtmpl': os.path.join(download_dir, 'temp_audio.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    print(f"🔍 Searching YouTube for: {query}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([query])
        except Exception as e:
            print(f"❌ Error downloading audio: {e}")
            return None

    # Trim the first 30 seconds using ffmpeg
    print(f"✂️ Trimming to 30 seconds...")
    temp_audio_path = os.path.join(download_dir, 'temp_audio.mp3')
    cmd = [
        "ffmpeg", "-y", "-i", temp_audio_path,
        "-t", "30", "-acodec", "copy", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Clean up temp file
    os.remove(temp_audio_path)

    print(f"✅ Saved 30-sec preview to: {output_path}")
    return output_path

# 🧪 Example usage
if __name__ == "__main__":
    download_30sec_audio_from_youtube("Radiohead Weird Fishes", download_dir="/Users/rileykrisch/Downloads/Personal Projects/Spectrogram/Youtube Previews/")
