# AUDIO_RAG Prototype

This repository contains a simple prototype of a voice agent that demonstrates
Retrieval Augmented Generation (RAG) from audio input.

## Components

- `frontend/` – React Native application with a single page to record audio and
  display the transcription and response from the backend.
- `backend/` – FastAPI service that handles speech-to-text, vector search via
  ChromaDB, and LLM interaction.

## Setup

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

Requires Node.js and Expo CLI.

```bash
cd frontend
npm install
npm start
```
After running `npm start`, open `http://localhost:19006` in a browser to access the web version. Alternatively, use the Expo Go mobile app to scan the QR code displayed in the terminal.

The mobile app expects the backend to run on `http://localhost:8000`.

## Sample Knowledge Base

Two markdown files in `backend/data/sample_kb/` are loaded into ChromaDB on
startup. Modify or add documents there to experiment with retrieval.

## Report

See [`report.md`](report.md) for a short comparison of latency, quality, and
cost considerations for different backend models.
