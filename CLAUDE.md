# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Streamlit web app for medical dictation transcription using Google's MedASR model (`google/medasr`). Supports audio upload and live recording with CTC beam search decoding. Optionally computes WER metrics against a reference transcript (pasted or uploaded as a `.txt` file) and offers JSON download of results.

## Commands

```bash
uv sync                                # Install all dependencies (including dev)
uv run streamlit run streamlit_app.py  # Run app
uv run pytest                          # Run tests
uv run ruff check .                    # Lint
uv run ruff format .                   # Format
uv run ty check                        # Type check
```

## Architecture

Audio input → librosa (16kHz) → AutoProcessor → model logits → log softmax → CTC beam search (kenlm, beam width 8) → transcribed text

- `streamlit_app.py` — Main app with model caching (`@st.cache_resource`), auto device detection (MPS > CUDA > CPU), custom CTC decoder, session state persistence, status feedback, WER metrics (metric + dataframe layout) with HTML word diff, reference transcript file upload (UTF-8 validated), and JSON download
- `utils/helper.py` — Text normalization, WER computation (jiwer), colored diff output (ANSI and HTML)
- `tests/test_helper.py` — Unit tests for helper utilities (normalize, compute_wer, colored_diff, html_diff, evaluate)
- `tests/test_app.py` — Unit tests for transcribe, _wer_label, audio_tab file upload (including empty and non-UTF-8 edge cases), and show_results layout with mocked model/audio/Streamlit
- `tests/data/` — Sample audio and reference transcripts

## Notes

- Unigram warnings from pyctcdecode are expected and suppressed — the MedASR tokenizer vocab does not align with the kenlm model's word vocabulary, matching the official notebook implementation
- `warnings.filterwarnings` calls must remain before library imports to suppress warnings at import time; imports use `# noqa: E402` to satisfy ruff
- On CUDA, inference runs in float16 for ~40-50% VRAM reduction and ~20-30% speedup; MPS and CPU remain float32 (float16 is unreliable on MPS, slower on CPU)
- `torch.compile` is applied on CUDA only — first inference is slower (compilation warmup), subsequent calls are ~10-30% faster

## Environment

- `HF_TOKEN` in `.env` for HuggingFace Hub authentication
- Model ID hardcoded as `google/medasr`
