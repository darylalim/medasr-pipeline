# MedASR Pipeline

Medical dictation transcription using Google's MedASR speech-to-text model.

## Features

- Upload audio files (wav, mp3, flac, ogg, webm, m4a) or record live
- CTC beam search decoding with kenlm language model
- WER evaluation metrics with qualitative labels and HTML word diff
- Session state persistence — results survive widget interactions
- Status feedback during model loading and transcription
- JSON download of transcription results

## Setup

```bash
uv sync
```

Create a `.env` file with your HuggingFace token:

```
HF_TOKEN=your_token_here
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

## Development

```bash
uv run pytest                # Run tests
uv run ruff check .          # Lint
uv run ruff format .         # Format
uv run ty check              # Type check
```
