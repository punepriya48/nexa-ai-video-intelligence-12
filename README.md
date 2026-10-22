
# Nexa AI — Video Intelligence & Meeting Assistant

An AI-powered application that transforms video and audio recordings into searchable, actionable insights.

## Features

- Download and process audio from YouTube URLs
- Upload supported audio and video files
- Transcribe speech using OpenAI Whisper
- Generate titles and summaries using Groq
- Extract action items, key decisions, and open questions
- Ask questions about recordings using Retrieval-Augmented Generation (RAG)
- Download reports and transcripts
- Modern Streamlit dashboard

## Tech Stack

- Python
- Streamlit
- OpenAI Whisper
- Groq API
- LangChain
- ChromaDB
- Hugging Face sentence-transformer embeddings
- yt-dlp
- FFmpeg

## Project Structure

```text
Video_Agent/
├── core/
│   ├── extractor.py
│   ├── rag_engine.py
│   ├── summarizer.py
│   ├── transcriber.py
│   └── vector_store.py
├── utils/
│   └── audio_processor.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

1. Install Python and Git.
2. Install FFmpeg and ensure it is available in your system PATH.
3. Clone this repository.
4. Open a terminal in the project folder.
5. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

6. Install the dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

7. Create your local environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

8. Open `.env` and add your own Groq API key. Never publish this file.

9. Start the application:

   ```powershell
   streamlit run main.py
   ```

## How It Works

1. The user provides a YouTube URL or uploads a recording.
2. The application processes and chunks the audio.
3. Whisper generates a transcript.
4. Groq generates a title, summary, and structured insights.
5. LangChain and ChromaDB prepare the transcript for question answering.
6. The user explores the results in the dashboard and downloads reports.

## Security

- Keep API keys in `.env`.
- Never commit API keys, private recordings, or other confidential data.
- Follow the terms and applicable rights for content you process.

## Future Improvements

- Meeting history and database storage
- Improved transcription for multilingual recordings
- Better handling of API rate limits
- User authentication and deployment
- Additional analytics and export formats

## Author

Developed as an AI-powered video and meeting intelligence project.
