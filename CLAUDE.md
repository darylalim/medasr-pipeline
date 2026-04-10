# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Streamlit web app for medical dictation transcription using Google's MedASR model (`google/medasr`). Two tabs: record live or upload audio. Supports CTC beam search decoding with transcription output in a text panel.

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

- `streamlit_app.py` — Main app: model caching (`@st.cache_resource`), device detection (CUDA > MPS, no CPU fallback), CTC decoder, two tabs (Record, Upload), transcription output in disabled text area
- `tests/test_app.py` — Tests for transcribe, _patch_feature_extractor, audio_tab

## Notes

- Unigram warnings from pyctcdecode are expected and suppressed — the MedASR tokenizer vocab does not align with the kenlm model's word vocabulary, matching the official notebook implementation
- `warnings.filterwarnings` calls must remain before library imports to suppress warnings at import time; imports use `# noqa: E402` to satisfy ruff
- On CUDA, inference runs in float16 for ~40-50% VRAM reduction and ~20-30% speedup; MPS remains float32 (float16 is unreliable on MPS)
- `torch.compile` is applied on CUDA only — first inference is slower (compilation warmup), subsequent calls are ~10-30% faster
- `_patch_feature_extractor` works around a transformers 5.2.0 bug (`huggingface/transformers#38341`) — remove when the upstream fix is released

## Environment

- `HF_TOKEN` in `.env` for HuggingFace Hub authentication
- Model ID hardcoded as `google/medasr`
