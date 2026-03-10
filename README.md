# MedASR Pipeline

Medical dictation transcription using Google's MedASR speech-to-text model.

## Features

- Upload audio files (wav, mp3, flac, ogg, webm, m4a) or record live
- CTC beam search decoding with kenlm language model
- Copy-to-clipboard transcription output
- Corrected transcript editing with JSON export
- WER evaluation with qualitative labels and HTML word diff
- Reference transcript input via text area or UTF-8 `.txt` file upload
- "Try with sample" demo button with bundled audio and reference
- Batch evaluation tab for multi-file transcription with aggregate WER
- Session state persistence — results survive widget interactions
- JSON download of transcription and batch results

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
