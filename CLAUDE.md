# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Streamlit web app for medical dictation transcription using Google's MedASR model (`google/medasr`). Three tabs: upload audio, record live, or batch evaluate multiple files. Supports CTC beam search decoding, WER evaluation against reference transcripts, corrected transcript editing, and JSON export.

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

- `streamlit_app.py` — Main app: model caching (`@st.cache_resource`), device detection (MPS > CUDA > CPU), CTC decoder, three tabs (Upload, Record, Batch Evaluate), `st.code` output with copy-to-clipboard, corrected transcript editing, sample demo button, WER metrics with HTML word diff, batch evaluation with aggregate WER and per-file expandable results, JSON download
- `utils/helper.py` — Text normalization, WER computation (jiwer), colored diff output (ANSI and HTML)
- `samples/` — Demo audio and reference transcript for the "Try with sample" button
- `tests/test_app.py` — Tests for transcribe, _wer_label, audio_tab (file upload, sample button, stale check), show_results (st.code, corrected transcript, JSON export, word diff), batch helpers (_match_refs, _aggregate_wer), and batch_tab (transcription, aggregate row, warnings, JSON download)
- `tests/test_helper.py` — Tests for normalize, compute_wer, colored_diff, html_diff, evaluate

## Notes

- Unigram warnings from pyctcdecode are expected and suppressed — the MedASR tokenizer vocab does not align with the kenlm model's word vocabulary, matching the official notebook implementation
- `warnings.filterwarnings` calls must remain before library imports to suppress warnings at import time; imports use `# noqa: E402` to satisfy ruff
- On CUDA, inference runs in float16 for ~40-50% VRAM reduction and ~20-30% speedup; MPS and CPU remain float32 (float16 is unreliable on MPS, slower on CPU)
- `torch.compile` is applied on CUDA only — first inference is slower (compilation warmup), subsequent calls are ~10-30% faster
- `_patch_feature_extractor` works around a transformers 5.2.0 bug (`huggingface/transformers#38341`) — remove when the upstream fix is released

## Environment

- `HF_TOKEN` in `.env` for HuggingFace Hub authentication
- Model ID hardcoded as `google/medasr`
