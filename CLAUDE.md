# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Streamlit web app for medical dictation transcription using Google's MedASR model (`google/medasr`). Supports audio upload and live recording with CTC beam search decoding. Optionally computes WER metrics against a reference transcript and offers JSON download of results.

## Commands

```bash
pip install -r requirements.txt   # Install dependencies
streamlit run streamlit_app.py    # Run app
pytest                            # Run tests
ruff check .                      # Lint
ruff format .                     # Format
ty check                          # Type check
```

## Architecture

Audio input → librosa (16kHz) → AutoProcessor → model logits → log softmax → CTC beam search (kenlm, beam width 8) → transcribed text

- `streamlit_app.py` — Main app with model caching (`@st.cache_resource`), auto device detection (MPS > CUDA > CPU), custom CTC decoder, metrics display, and JSON download
- `utils/helper.py` — Text normalization, WER computation (jiwer), colored diff output
- `tests/test_helper.py` — Unit tests for helper utilities (normalize, compute_wer, colored_diff, evaluate)
- `tests/test_app.py` — Unit tests for transcribe function with mocked model/audio
- `tests/data/` — Sample audio and reference transcripts

## Notes

- Unigram warnings from pyctcdecode are expected and suppressed — the MedASR tokenizer vocab does not align with the kenlm model's word vocabulary, matching the official notebook implementation
- `warnings.filterwarnings` calls must remain before library imports to suppress warnings at import time; imports use `# noqa: E402` to satisfy ruff

## Environment

- `HF_TOKEN` in `.env` for HuggingFace Hub authentication
- Model ID hardcoded as `google/medasr`
