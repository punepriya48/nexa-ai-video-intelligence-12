
import os
from pathlib import Path

import yt_dlp
from pydub import AudioSegment


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = BASE_DIR / "downloades"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """Download YouTube audio and convert it to WAV format."""

    output_path = str(DOWNLOAD_DIR / "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    # FFmpegExtractAudio changes the downloaded extension to WAV.
    wav_path = str(Path(filename).with_suffix(".wav"))

    if not os.path.isfile(wav_path):
        raise FileNotFoundError(
            f"Downloaded WAV file was not found: {wav_path}"
        )

    print(f"Audio downloaded successfully: {wav_path}")
    return wav_path


def convert_to_wav(input_path: str) -> str:
    """Convert a local audio/video file to mono, 16 kHz WAV."""

    input_file = Path(input_path)

    if not input_file.is_file():
        raise FileNotFoundError(
            f"Input file was not found: {input_path}"
        )

    output_path = input_file.with_name(
        input_file.stem + "_converted.wav"
    )

    audio = AudioSegment.from_file(str(input_file))
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(str(output_path), format="wav")

    print(f"Converted audio saved: {output_path}")
    return str(output_path)


def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list[str]:
    """Split a WAV file into smaller WAV chunks."""

    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be greater than zero.")

    audio_file = Path(wav_path)

    if not audio_file.is_file():
        raise FileNotFoundError(
            f"WAV file was not found: {wav_path}"
        )

    audio = AudioSegment.from_wav(str(audio_file))
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]

        chunk_path = audio_file.with_name(
            f"{audio_file.stem}_chunk_{i}.wav"
        )

        chunk.export(str(chunk_path), format="wav")
        chunks.append(str(chunk_path))

    print(f"Audio ready: {len(chunks)} chunk(s) created.")
    return chunks


def process_input(source: str) -> list[str]:
    """Accept a YouTube URL or local file and return WAV chunks."""

    if not source or not source.strip():
        raise ValueError("Please provide a YouTube URL or file path.")

    source = source.strip()

    try:
        if source.startswith(("http://", "https://")):
            print("Detected YouTube URL. Downloading audio...")
            wav_path = download_youtube_audio(source)
        else:
            print("Detected local file. Converting to WAV...")
            wav_path = convert_to_wav(source)

        print("Chunking audio...")
        chunks = chunk_audio(wav_path)

        return chunks

    except Exception as exc:
        print(f"Audio processing failed: {exc}")
        raise
