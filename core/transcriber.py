
import os
import requests
import whisper
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

# Configuration
TRANSCRIBER = os.getenv("TRANSCRIBER", "whisper").strip().lower()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small").strip()
SARVAM_PIECE_SECONDS = 25

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = (
    "https://api.sarvam.ai/speech-to-text-translate"
)
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None


def load_model():
    """Load the Whisper model once and reuse it."""
    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")

    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    """Transcribe one audio chunk using Whisper."""
    if not os.path.isfile(chunk_path):
        raise FileNotFoundError(f"Audio file not found: {chunk_path}")

    model = load_model()

    result = model.transcribe(
        chunk_path,
        task="transcribe",
        language="en",
        fp16=False,
    )

    return result.get("text", "").strip()


def _send_to_sarvam(piece_path: str) -> str:
    """Send one short WAV file to Sarvam and return its transcript."""
    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is missing from your .env file."
        )

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    with open(piece_path, "rb") as audio_file:
        files = {
            "file": (
                os.path.basename(piece_path),
                audio_file,
                "audio/wav",
            )
        }

        data = {
            "model": SARVAM_MODEL,
            "with_diarization": "false",
        }

        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"Sarvam API error: {response.status_code}")
        print(response.text)
        response.raise_for_status()

    result = response.json()
    return result.get("transcript", "").strip()


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Split an audio chunk into short WAV pieces and translate
    the speech to English using Sarvam.
    """
    if not os.path.isfile(chunk_path):
        raise FileNotFoundError(f"Audio file not found: {chunk_path}")

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is missing from your .env file."
        )

    audio = AudioSegment.from_file(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    transcripts = []
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start:start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"

        try:
            piece.export(piece_path, format="wav")

            print(
                f"  Sarvam piece {i + 1}/{total_pieces} ..."
            )

            text = _send_to_sarvam(piece_path)

            if text:
                transcripts.append(text)

        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(transcripts).strip()


def transcribe_chunk(
    chunk_path: str,
    language: str = "english",
) -> str:
    """
    Select the transcription engine.

    English  -> Whisper
    Hinglish -> Sarvam speech-to-text translation
    """
    language = language.strip().lower()

    if language == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)

    if TRANSCRIBER == "sarvam":
        return transcribe_chunk_sarvam(chunk_path)

    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(
    chunks: list,
    language: str = "english",
) -> str:
    """Transcribe all audio chunks and combine their text."""
    if not chunks:
        print("No audio chunks received.")
        return ""

    engine = (
        "Sarvam AI"
        if language.strip().lower() == "hinglish"
        or TRANSCRIBER == "sarvam"
        else "Whisper"
    )

    print(f"Using {engine} for transcription.")

    transcripts = []

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        try:
            text = transcribe_chunk(chunk, language=language)

            if text:
                transcripts.append(text)

        except Exception as error:
            print(
                f"Failed to transcribe chunk {i + 1}: {error}"
            )
            raise

    print("Transcription complete.")

    return " ".join(transcripts).strip()
