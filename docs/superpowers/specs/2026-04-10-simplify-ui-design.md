# Simplify UI for Core Pipeline

**Date:** 2026-04-10
**Status:** Approved

## Goal

Strip the Streamlit app down to a focused transcription tool — remove evaluation, batch processing, and demo features to prepare it as a core pipeline for a medical dictation product.

## What Stays

- Model loading (`load_model`), device detection (MPS > CUDA > CPU), CTC decoder setup
- `_patch_feature_extractor()` (transformers 5.2.0 workaround)
- `transcribe()` function (audio bytes in, text out)
- Two tabs: **Record** (first), **Upload** (second)
- Transcription output displayed in a `st.text_area` with `disabled=True`

## What's Removed

### Features
- **Batch Evaluate tab** — `batch_tab()`, `_match_refs()`, `_aggregate_wer()`
- **Reference transcript comparison** — WER evaluation, reference file upload, reference text area, word diff display
- **Sample demo button** — "Try with sample" button, `SAMPLE_DIR`/`SAMPLE_AUDIO`/`SAMPLE_REF` constants, related session state keys
- **Corrected transcript editor** — `st.text_area` for corrections
- **JSON download** — `st.download_button` for results export
- **`show_results()`** — replaced by inline `st.text_area(disabled=True)`
- **`_wer_label()`** — no longer needed

### Files
- **`utils/helper.py`** — all functions (`normalize`, `compute_wer`, `colored_diff`, `html_diff`, `evaluate`) become unused; delete entire file
- **`tests/test_helper.py`** — tests for removed helper functions; delete entire file
- **`samples/sample_audio.wav`** and **`samples/sample_transcript.txt`** — no longer referenced; delete directory

### Dependencies (from `pyproject.toml`)
- `jiwer`
- `levenshtein`

### Tests (from `tests/test_app.py`)
- `TestShowResults` — entire class
- `TestAudioTabSample` — entire class
- `TestMatchRefs` — entire class
- `TestAggregateWer` — entire class
- `TestBatchTab` — entire class
- `TestAudioTabFileUpload` — remove reference-transcript-related tests
- `_mock_expander`, `_mock_batch_uploaders` helpers — if no longer used

## What's Changed

| Item | Before | After |
|------|--------|-------|
| Page title | "Medical Dictation Transcription" | "MedASR Pipeline" |
| Tab labels | "Upload Audio", "Record Audio" | "Upload", "Record" |
| Tab order | Upload first | Record first |
| File uploader label | "Upload audio file" | Hidden (empty string with `label_visibility="collapsed"`) |
| Record audio label | "Record audio" | Hidden (empty string with `label_visibility="collapsed"`) |
| Accepted audio types | `wav, mp3, flac, ogg, webm, m4a` | `flac, m4a, mp3, ogg, wav, webm` (alphabetical) |
| Transcription output | `st.code(text, language=None)` | `st.text_area(value=text, disabled=True)` |

## Resulting App Structure

```
streamlit_app.py    # Model loading, transcribe(), two tabs (Record, Upload)
pyproject.toml      # Dependencies (minus jiwer, levenshtein)
tests/test_app.py   # Tests for transcribe, _patch_feature_extractor, audio_tab
```

## CLAUDE.md Updates

Update the overview, architecture description, and notes to reflect the simplified app.
