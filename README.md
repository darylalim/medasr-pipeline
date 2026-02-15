# MedASR Pipeline

Medical dictation transcription using Google's MedASR speech-to-text model.

## Features

- Upload audio files (wav, mp3, flac, ogg, webm, m4a) or record live
- CTC beam search decoding with kenlm language model
- WER evaluation metrics when a reference transcript is provided
- JSON download of transcription results

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your HuggingFace token:

```
HF_TOKEN=your_token_here
```

## Usage

```bash
streamlit run streamlit_app.py
```

## Development

```bash
pytest                # Run tests
ruff check .          # Lint
ruff format .         # Format
ty check              # Type check
```
