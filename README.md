# MedASR Pipeline

Medical dictation transcription using Google's MedASR speech-to-text model.

## Features

- Record live audio or upload files (flac, m4a, mp3, ogg, wav, webm)
- CTC beam search decoding with kenlm language model
- GPU-accelerated inference (CUDA > MPS)
- Download transcription as text file

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
