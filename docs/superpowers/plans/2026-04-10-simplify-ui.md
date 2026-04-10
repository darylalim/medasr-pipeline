# Simplify UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip the app to a focused transcription tool — two tabs (Record, Upload), no evaluation, no batch, no demo features.

**Architecture:** Keep the existing model loading and transcription pipeline. Replace the three-tab UI with two simplified tabs that just transcribe and display results in a disabled text area. Remove all evaluation, batch, sample, and export code along with their dependencies.

**Tech Stack:** Streamlit, PyTorch, Transformers, librosa, pyctcdecode, kenlm

**Spec:** `docs/superpowers/specs/2026-04-10-simplify-ui-design.md`

---

### Task 1: Delete unused files and directories

**Files:**
- Delete: `utils/helper.py`
- Delete: `utils/` (directory, after helper.py removed)
- Delete: `tests/test_helper.py`
- Delete: `samples/sample_audio.wav`
- Delete: `samples/sample_transcript.txt`
- Delete: `samples/` (directory)

- [ ] **Step 1: Delete the files and directories**

```bash
rm utils/helper.py
rmdir utils
rm tests/test_helper.py
rm samples/sample_audio.wav samples/sample_transcript.txt
rmdir samples
```

- [ ] **Step 2: Verify deletions**

Run: `ls utils/ samples/ tests/test_helper.py 2>&1`
Expected: all "No such file or directory" errors

- [ ] **Step 3: Commit**

```bash
git add -u utils/ samples/ tests/test_helper.py
git commit -m "Remove unused helper, samples, and helper tests"
```

---

### Task 2: Remove unused dependencies

**Files:**
- Modify: `pyproject.toml:6-17`

- [ ] **Step 1: Edit pyproject.toml to remove jiwer and levenshtein**

Replace the dependencies list in `pyproject.toml` (lines 6-17):

```toml
dependencies = [
    "huggingface-hub",
    "kenlm",
    "librosa",
    "pyctcdecode",
    "python-dotenv",
    "streamlit",
    "torch",
    "transformers",
]
```

- [ ] **Step 2: Run uv sync to update lockfile**

Run: `uv sync`
Expected: resolves without jiwer or levenshtein

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Remove jiwer and levenshtein dependencies"
```

---

### Task 3: Simplify streamlit_app.py

**Files:**
- Modify: `streamlit_app.py`

This task rewrites the app to its simplified form. The resulting file keeps: warning suppression, imports (minus json, helper imports), model loading, transcribe, `_patch_feature_extractor`, and a simplified `audio_tab` that just transcribes and displays a disabled text area.

- [ ] **Step 1: Remove unused imports**

Remove `import json` (line 2) and the `from utils.helper` import (line 21). Also remove `from pathlib import Path` (line 4) since `SAMPLE_DIR`/`SAMPLE_AUDIO`/`SAMPLE_REF` are being removed.

After edits, the import block (lines 1-21) becomes:

```python
import io
import warnings

for msg in [
    "Using padding='same' with even kernel lengths",
    "Unigrams not provided",
    "No known unigrams provided",
]:
    warnings.filterwarnings("ignore", message=msg)

from dotenv import load_dotenv  # noqa: E402
import huggingface_hub  # noqa: E402
import librosa  # noqa: E402
import pyctcdecode  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCTC, AutoProcessor  # noqa: E402
```

- [ ] **Step 2: Remove sample constants**

Delete lines 35-37 (`SAMPLE_DIR`, `SAMPLE_AUDIO`, `SAMPLE_REF`).

- [ ] **Step 3: Remove `_wer_label` and `show_results`**

Delete `_wer_label` (lines 87-94) and `show_results` (lines 97-149) entirely.

- [ ] **Step 4: Rewrite `audio_tab` to simplified version**

Replace the entire `audio_tab` function (lines 152-228) with:

```python
def audio_tab(audio_data, key: str) -> None:
    if audio_data is not None:
        st.audio(audio_data)

    if st.button("Transcribe", key=f"transcribe_{key}", disabled=(audio_data is None)):
        with st.status("Transcribing...", expanded=True) as status:
            st.write("Loading model...")
            processor, model, decoder = load_model()
            st.write("Transcribing audio...")
            text = transcribe(audio_data.getvalue(), processor, model, decoder)
            status.update(label="Complete!", state="complete", expanded=False)
        st.session_state[f"text_{key}"] = text
        st.toast("Transcription complete!")

    if f"text_{key}" in st.session_state:
        st.subheader("Transcription")
        st.text_area(
            "Transcription",
            value=st.session_state[f"text_{key}"],
            disabled=True,
            label_visibility="collapsed",
        )
```

- [ ] **Step 5: Remove all batch code**

Delete `_match_refs` (lines 231-240), `_aggregate_wer` (lines 243-253), and `batch_tab` (lines 256-354) entirely.

- [ ] **Step 6: Rewrite app entry section**

Replace lines 357-373 with:

```python
st.set_page_config(page_title="MedASR", page_icon="\U0001fa7a", layout="centered")
st.title("MedASR Pipeline")

tab_record, tab_upload = st.tabs(["Record", "Upload"])
with tab_record:
    audio_tab(
        st.audio_input("Record audio", label_visibility="collapsed"),
        "record",
    )
with tab_upload:
    audio_tab(
        st.file_uploader(
            "Upload audio file",
            type=["flac", "m4a", "mp3", "ogg", "wav", "webm"],
            label_visibility="collapsed",
        ),
        "upload",
    )
```

- [ ] **Step 7: Verify the complete file looks correct**

Run: `uv run ruff check streamlit_app.py && uv run ruff format --check streamlit_app.py`
Expected: no lint or format errors

- [ ] **Step 8: Commit**

```bash
git add streamlit_app.py
git commit -m "Simplify app: two tabs, no eval/batch/sample/export"
```

---

### Task 4: Simplify tests

**Files:**
- Modify: `tests/test_app.py`

- [ ] **Step 1: Rewrite test_app.py**

Remove all test classes and helpers except `TestTranscribe`, `TestPatchFeatureExtractor`, and `TestWerLabel` (which is removed — no longer exists). Remove the `_mock_expander`, `_mock_status`, `_mock_batch_uploaders` helpers. Remove the `json` import, `show_results`/`_wer_label`/`audio_tab`/`_match_refs`/`_aggregate_wer`/`batch_tab` from the import block.

Update the import block to:

```python
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from streamlit_app import (
    transcribe,
    _patch_feature_extractor,
    audio_tab,
)
```

Keep these test classes unchanged:
- `TestTranscribe` (lines 65-127) — all 6 tests
- `TestPatchFeatureExtractor` (lines 129-152) — both tests

Add a new simplified `TestAudioTab` class:

```python
class TestAudioTab:
    @patch("streamlit_app.st")
    def test_transcribe_button_disabled_without_audio(self, mock_st):
        mock_st.button.return_value = False
        mock_st.session_state = {}

        audio_tab(None, "upload")

        mock_st.button.assert_called_once_with(
            "Transcribe", key="transcribe_upload", disabled=True
        )

    @patch("streamlit_app.transcribe", return_value="hello world")
    @patch(
        "streamlit_app.load_model",
        return_value=(MagicMock(), MagicMock(), MagicMock()),
    )
    @patch("streamlit_app.st")
    def test_transcribe_stores_result(self, mock_st, mock_load_model, mock_transcribe):
        mock_st.button.return_value = True
        mock_st.status.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_st.status.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.session_state = {}
        audio_data = MagicMock()
        audio_data.getvalue.return_value = b"fake_audio"

        audio_tab(audio_data, "upload")

        mock_transcribe.assert_called_once()
        assert mock_st.session_state["text_upload"] == "hello world"

    @patch("streamlit_app.st")
    def test_displays_transcription_when_in_session(self, mock_st):
        mock_st.button.return_value = False
        mock_st.session_state = {"text_upload": "transcribed text"}

        audio_tab(None, "upload")

        mock_st.text_area.assert_called_once_with(
            "Transcription",
            value="transcribed text",
            disabled=True,
            label_visibility="collapsed",
        )
```

- [ ] **Step 2: Run all tests**

Run: `uv run pytest tests/test_app.py -v`
Expected: all tests pass (6 transcribe + 2 patch + 3 audio_tab = 11 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/test_app.py
git commit -m "Simplify tests: remove batch/eval/sample test classes"
```

---

### Task 5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Replace CLAUDE.md content**

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Streamlit web app for medical dictation transcription using Google's MedASR model (`google/medasr`). Two tabs: record live or upload audio. Supports CTC beam search decoding with transcription output in a text panel.

## Commands

` ` `bash
uv sync                                # Install all dependencies (including dev)
uv run streamlit run streamlit_app.py  # Run app
uv run pytest                          # Run tests
uv run ruff check .                    # Lint
uv run ruff format .                   # Format
uv run ty check                        # Type check
` ` `

## Architecture

Audio input → librosa (16kHz) → AutoProcessor → model logits → log softmax → CTC beam search (kenlm, beam width 8) → transcribed text

- `streamlit_app.py` — Main app: model caching (`@st.cache_resource`), device detection (MPS > CUDA > CPU), CTC decoder, two tabs (Record, Upload), transcription output in disabled text area
- `tests/test_app.py` — Tests for transcribe, _patch_feature_extractor, audio_tab

## Notes

- Unigram warnings from pyctcdecode are expected and suppressed — the MedASR tokenizer vocab does not align with the kenlm model's word vocabulary, matching the official notebook implementation
- `warnings.filterwarnings` calls must remain before library imports to suppress warnings at import time; imports use `# noqa: E402` to satisfy ruff
- On CUDA, inference runs in float16 for ~40-50% VRAM reduction and ~20-30% speedup; MPS and CPU remain float32 (float16 is unreliable on MPS, slower on CPU)
- `torch.compile` is applied on CUDA only — first inference is slower (compilation warmup), subsequent calls are ~10-30% faster
- `_patch_feature_extractor` works around a transformers 5.2.0 bug (`huggingface/transformers#38341`) — remove when the upstream fix is released

## Environment

- `HF_TOKEN` in `.env` for HuggingFace Hub authentication
- Model ID hardcoded as `google/medasr`
```

(Note: the triple backticks in the Commands section above are escaped for this plan — use real triple backticks in the actual file.)

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "Update CLAUDE.md for simplified app"
```

---

### Task 6: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: all 11 tests pass, no warnings about missing imports

- [ ] **Step 2: Run linter and formatter**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: no errors

- [ ] **Step 3: Verify no stale references**

Run: `grep -r "helper\|jiwer\|levenshtein\|batch_tab\|show_results\|_wer_label\|_match_refs\|_aggregate_wer\|SAMPLE_AUDIO\|SAMPLE_REF\|SAMPLE_DIR\|sample_audio\|sample_transcript" streamlit_app.py tests/ 2>/dev/null`
Expected: no matches

- [ ] **Step 4: Commit if any fixups were needed**

Only if previous steps required changes:
```bash
git add -A
git commit -m "Fix lint/test issues from simplification"
```
