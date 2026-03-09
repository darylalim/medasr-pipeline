# UI Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add sample audio demo, copy-to-clipboard, corrected transcript editing, and batch evaluation to the MedASR Streamlit app.

**Architecture:** Four additive changes to `streamlit_app.py` — `show_results` gets `st.code` + corrected transcript field, `audio_tab` gets a sample button backed by a `samples/` directory, and a new `batch_tab` function provides multi-file evaluation with aggregate WER in a third tab.

**Tech Stack:** Streamlit, pathlib, existing `utils/helper.py` functions

---

### Task 1: Create samples/ directory

**Files:**
- Create: `samples/sample_audio.wav` (copy from `tests/data/audio/test_audio.wav`)
- Create: `samples/sample_transcript.txt` (copy from `tests/data/text/sample_transcript.txt`)

**Step 1: Copy sample files**

```bash
mkdir -p samples
cp tests/data/audio/test_audio.wav samples/sample_audio.wav
cp tests/data/text/sample_transcript.txt samples/sample_transcript.txt
```

**Step 2: Commit**

```bash
git add samples/
git commit -m "Add samples/ directory with demo audio and transcript"
```

---

### Task 2: show_results — st.code + corrected transcript

**Files:**
- Modify: `streamlit_app.py:91-133` (`show_results` function)
- Modify: `tests/test_app.py` (`TestShowResults` class + imports)

**Step 1: Write failing tests**

Add `import json` to top of `tests/test_app.py`.

Add to `TestShowResults` class:

```python
@patch("streamlit_app.st")
def test_uses_st_code_for_transcription(self, mock_st):
    mock_st.text_area.return_value = "hello world"
    show_results("hello world", "", "test")
    mock_st.code.assert_called_once_with("hello world", language=None)

@patch("streamlit_app.st")
def test_corrected_transcript_shown(self, mock_st):
    mock_st.text_area.return_value = "hello world"
    show_results("hello world", "", "test")
    mock_st.text_area.assert_called_once_with(
        "Corrected transcript",
        value="hello world",
        height=200,
        key="corrected_test",
    )

@patch("streamlit_app.st")
def test_json_excludes_corrected_when_unchanged(self, mock_st):
    mock_st.text_area.return_value = "hello world"
    show_results("hello world", "", "test")
    data = json.loads(mock_st.download_button.call_args[0][1])
    assert "corrected_transcription" not in data

@patch("streamlit_app.st")
def test_json_includes_corrected_when_changed(self, mock_st):
    mock_st.text_area.return_value = "corrected text"
    show_results("hello world", "", "test")
    data = json.loads(mock_st.download_button.call_args[0][1])
    assert data["corrected_transcription"] == "corrected text"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_app.py::TestShowResults -v`
Expected: 4 new tests FAIL (no `st.code` call, no corrected transcript logic)

**Step 3: Implement show_results changes**

Replace the `show_results` function in `streamlit_app.py` (lines 91-133):

```python
def show_results(text: str, ref_text: str, key: str):
    st.subheader("Transcription")
    st.code(text, language=None)

    corrected = st.text_area(
        "Corrected transcript",
        value=text,
        height=200,
        key=f"corrected_{key}",
    )

    result = {"transcription": text}
    if corrected != text:
        result["corrected_transcription"] = corrected

    if ref_text.strip():
        metrics = compute_wer(ref_text, text)
        result["reference"] = ref_text
        result["metrics"] = metrics
        ref_tokens = metrics["ref_tokens"]

        st.subheader("Evaluation Metrics")
        col_wer, col_breakdown = st.columns(2)
        col_wer.metric(
            "WER",
            f"{metrics['wer'] * 100:.2f}%",
            delta=_wer_label(metrics["wer"]),
            delta_color="off",
        )
        col_breakdown.dataframe(
            {
                "Type": ["Insertions", "Deletions", "Substitutions"],
                "Count": [
                    metrics["insertions"],
                    metrics["deletions"],
                    metrics["substitutions"],
                ],
                "Reference Words": [ref_tokens] * 3,
            },
            hide_index=True,
            width="stretch",
        )

        st.subheader("Word-Level Diff")
        st.markdown(html_diff(ref_text, text), unsafe_allow_html=True)

    st.download_button(
        "Download JSON",
        json.dumps(result, indent=2),
        "transcription_result.json",
        "application/json",
        key=f"dl_{key}",
    )
```

**Step 4: Fix existing tests**

The corrected transcript `st.text_area` call returns a `MagicMock` by default, which isn't JSON-serializable. Add `mock_st.text_area.return_value` to each existing `TestShowResults` test to match the `text` argument:

- `test_uses_two_column_layout`: add `mock_st.text_area.return_value = "hello world"`
- `test_wer_metric_in_left_column`: add `mock_st.text_area.return_value = "hello world"`
- `test_dataframe_in_right_column`: add `mock_st.text_area.return_value = "the cat sat"`
- `test_no_metrics_without_reference`: add `mock_st.text_area.return_value = "hello world"`

**Step 5: Run all tests**

Run: `uv run pytest tests/test_app.py::TestShowResults -v`
Expected: all 8 tests PASS

**Step 6: Commit**

```bash
git add streamlit_app.py tests/test_app.py
git commit -m "Replace text_area with st.code and add corrected transcript field"
```

---

### Task 3: Sample button in audio_tab

**Files:**
- Modify: `streamlit_app.py` (imports, constants, `audio_tab` function)
- Modify: `tests/test_app.py` (`TestAudioTabSample` class + imports)

**Step 1: Write failing tests**

Add new test class to `tests/test_app.py`:

```python
class TestAudioTabSample:
    @patch("streamlit_app.SAMPLE_AUDIO")
    @patch("streamlit_app.st")
    def test_sample_button_shown_when_files_exist(self, mock_st, mock_sample_audio):
        mock_sample_audio.exists.return_value = True
        mock_st.button.return_value = False
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.file_uploader.return_value = None
        mock_st.text_area.return_value = ""
        mock_st.session_state = {}

        audio_tab(None, "upload")

        # First button call is "Try with sample"
        first_call = mock_st.button.call_args_list[0]
        assert first_call[0][0] == "Try with sample"

    @patch("streamlit_app.SAMPLE_AUDIO")
    @patch("streamlit_app.st")
    def test_sample_button_hidden_when_no_files(self, mock_st, mock_sample_audio):
        mock_sample_audio.exists.return_value = False
        mock_st.button.return_value = False
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.file_uploader.return_value = None
        mock_st.text_area.return_value = ""
        mock_st.session_state = {}

        audio_tab(None, "upload")

        # Only the Transcribe button, no sample button
        assert mock_st.button.call_count == 1
        assert mock_st.button.call_args[0][0] == "Transcribe"

    @patch("streamlit_app.show_results")
    @patch("streamlit_app.transcribe", return_value="sample transcription")
    @patch("streamlit_app.load_model", return_value=(MagicMock(), MagicMock(), MagicMock()))
    @patch("streamlit_app.SAMPLE_REF")
    @patch("streamlit_app.SAMPLE_AUDIO")
    @patch("streamlit_app.st")
    def test_sample_click_transcribes_and_stores(
        self, mock_st, mock_sample_audio, mock_sample_ref,
        mock_load_model, mock_transcribe, mock_show_results,
    ):
        mock_sample_audio.exists.return_value = True
        mock_sample_audio.read_bytes.return_value = b"sample_wav"
        mock_sample_ref.read_text.return_value = "sample ref"
        # First button (Try with sample) = True, second (Transcribe) = False
        mock_st.button.side_effect = [True, False]
        mock_st.status.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_st.status.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.file_uploader.return_value = None
        mock_st.text_area.return_value = ""
        mock_st.session_state = {}

        audio_tab(None, "upload")

        mock_transcribe.assert_called_once()
        assert mock_st.session_state["text_upload"] == "sample transcription"
        assert mock_st.session_state["sample_ref_upload"] == "sample ref"
        assert mock_st.session_state["audio_id_upload"] == "sample"

    @patch("streamlit_app.show_results")
    @patch("streamlit_app.SAMPLE_AUDIO")
    @patch("streamlit_app.st")
    def test_sample_ref_used_as_fallback(self, mock_st, mock_sample_audio, mock_show_results):
        mock_sample_audio.exists.return_value = True
        mock_sample_audio.read_bytes.return_value = b"sample_wav"
        mock_st.button.return_value = False
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.file_uploader.return_value = None
        mock_st.text_area.return_value = ""
        mock_st.session_state = {
            "audio_id_upload": "sample",
            "text_upload": "transcribed",
            "sample_ref_upload": "sample ref text",
        }

        audio_tab(None, "upload")

        mock_show_results.assert_called_once_with("transcribed", "sample ref text", "upload")
```

Update imports at top of test file:

```python
from streamlit_app import transcribe, _wer_label, audio_tab, show_results
```

(No new imports needed from streamlit_app — `SAMPLE_AUDIO`, `SAMPLE_REF` are patched.)

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_app.py::TestAudioTabSample -v`
Expected: FAIL — `SAMPLE_AUDIO` not defined

**Step 3: Add imports and constants**

Add to `streamlit_app.py` after the existing imports (after line 20):

```python
from pathlib import Path  # noqa: E402
```

Add after `DTYPE` constant (after line 32):

```python
SAMPLE_DIR = Path(__file__).parent / "samples"
SAMPLE_AUDIO = SAMPLE_DIR / "sample_audio.wav"
SAMPLE_REF = SAMPLE_DIR / "sample_transcript.txt"
```

**Step 4: Implement audio_tab changes**

Replace the `audio_tab` function in `streamlit_app.py`:

```python
def audio_tab(audio_data, key: str):
    if SAMPLE_AUDIO.exists():
        if st.button("Try with sample", key=f"sample_{key}"):
            with st.status("Transcribing sample...", expanded=True) as status:
                st.write("Loading model...")
                processor, model, decoder = load_model()
                st.write("Transcribing audio...")
                text = transcribe(SAMPLE_AUDIO.read_bytes(), processor, model, decoder)
                status.update(label="Complete!", state="complete", expanded=False)
            st.session_state[f"text_{key}"] = text
            st.session_state[f"audio_id_{key}"] = "sample"
            st.session_state[f"sample_ref_{key}"] = SAMPLE_REF.read_text()
            st.toast("Transcription complete!")

    with st.expander("Compare against reference transcript"):
        ref_file = st.file_uploader(
            "Upload reference transcript",
            type=["txt"],
            key=f"ref_file_{key}",
        )
        ref = st.text_area(
            "Or paste reference transcript",
            key=f"ref_{key}",
            height=100,
            help="Paste the expected ground truth text to compute word error rate (WER) metrics against the transcription.",
        )
        if ref_file is not None:
            try:
                ref = ref_file.getvalue().decode("utf-8")
                if ref.strip():
                    st.caption("Using uploaded file.")
            except UnicodeDecodeError:
                st.error(
                    "Could not read file. Please upload a UTF-8 encoded text file."
                )
                ref = ""

    # Fallback to sample ref when no user-provided ref
    if not ref.strip() and ref_file is None and f"sample_ref_{key}" in st.session_state:
        ref = st.session_state[f"sample_ref_{key}"]

    if audio_data is not None:
        st.audio(audio_data)
    elif st.session_state.get(f"audio_id_{key}") == "sample":
        st.audio(SAMPLE_AUDIO.read_bytes(), format="audio/wav")

    # Clear stale results when audio changes
    if audio_data is not None:
        audio_id = audio_data.size
    elif st.session_state.get(f"audio_id_{key}") == "sample":
        audio_id = "sample"
    else:
        audio_id = None

    if st.session_state.get(f"audio_id_{key}") != audio_id:
        st.session_state[f"audio_id_{key}"] = audio_id
        st.session_state.pop(f"text_{key}", None)
        st.session_state.pop(f"sample_ref_{key}", None)

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
        show_results(st.session_state[f"text_{key}"], ref, key)
```

**Step 5: Fix existing TestAudioTabFileUpload tests**

The existing `_setup_st` helper needs to account for the new sample button. The mock_st.button now gets called twice (sample + transcribe). Update `_setup_st` to handle both buttons:

Since `SAMPLE_AUDIO.exists()` is checked, and `SAMPLE_AUDIO` is a real `Path` object in the test context (not mocked in these tests), it will check the actual filesystem. If `samples/sample_audio.wav` exists (from Task 1), the sample button renders. We need to either:
- Patch `SAMPLE_AUDIO` in these tests too, OR
- Set `mock_st.button.side_effect` to return `[False, False]` (sample=False, transcribe=False)

The simplest fix: patch `SAMPLE_AUDIO` to not exist:

```python
@patch("streamlit_app.show_results")
@patch("streamlit_app.SAMPLE_AUDIO")
@patch("streamlit_app.st")
def test_file_upload_overrides_text_area(self, mock_st, mock_sample_audio, mock_show_results):
    mock_sample_audio.exists.return_value = False
    # ...rest unchanged...
```

Apply this pattern to all 4 existing `TestAudioTabFileUpload` tests.

**Step 6: Run all tests**

Run: `uv run pytest tests/test_app.py -v`
Expected: all tests PASS

**Step 7: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add streamlit_app.py tests/test_app.py
git commit -m "Add sample audio button and reference transcript fallback"
```

---

### Task 4: Batch evaluation helpers

**Files:**
- Modify: `streamlit_app.py` (add `_match_refs`, `_aggregate_wer`)
- Modify: `tests/test_app.py` (add `TestMatchRefs`, `TestAggregateWer`)

**Step 1: Write failing tests**

Update imports in `tests/test_app.py`:

```python
from streamlit_app import (
    transcribe, _wer_label, audio_tab, show_results,
    _match_refs, _aggregate_wer,
)
```

Add test classes:

```python
class TestMatchRefs:
    def test_matches_by_stem(self):
        f1 = MagicMock()
        f1.name = "patient1.txt"
        f1.getvalue.return_value = b"reference text"

        ref_map, errors = _match_refs([f1])
        assert ref_map == {"patient1": "reference text"}
        assert errors == []

    def test_skips_non_utf8(self):
        f1 = MagicMock()
        f1.name = "bad.txt"
        f1.getvalue.return_value = b"\xff\xfe"

        ref_map, errors = _match_refs([f1])
        assert ref_map == {}
        assert errors == ["bad.txt"]

    def test_empty_list(self):
        ref_map, errors = _match_refs([])
        assert ref_map == {}
        assert errors == []


class TestAggregateWer:
    def test_basic_aggregation(self):
        results = [
            {"metrics": {"insertions": 1, "deletions": 0, "substitutions": 0, "ref_tokens": 10, "wer": 0.1}},
            {"metrics": {"insertions": 0, "deletions": 1, "substitutions": 1, "ref_tokens": 10, "wer": 0.2}},
        ]
        assert _aggregate_wer(results) == 3 / 20

    def test_no_metrics_returns_none(self):
        results = [{"filename": "test.wav", "transcription": "hello"}]
        assert _aggregate_wer(results) is None

    def test_mixed_with_and_without_metrics(self):
        results = [
            {"metrics": {"insertions": 2, "deletions": 0, "substitutions": 0, "ref_tokens": 10, "wer": 0.2}},
            {"filename": "no_ref.wav", "transcription": "hello"},
        ]
        assert _aggregate_wer(results) == 2 / 10
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_app.py::TestMatchRefs tests/test_app.py::TestAggregateWer -v`
Expected: ImportError — `_match_refs`, `_aggregate_wer` not defined

**Step 3: Implement helpers**

Add to `streamlit_app.py` after the `audio_tab` function:

```python
def _match_refs(ref_files):
    ref_map = {}
    errors = []
    for f in ref_files:
        stem = Path(f.name).stem
        try:
            ref_map[stem] = f.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f.name)
    return ref_map, errors


def _aggregate_wer(results):
    total_edits = 0
    total_ref_tokens = 0
    for r in results:
        if "metrics" in r:
            m = r["metrics"]
            total_edits += m["insertions"] + m["deletions"] + m["substitutions"]
            total_ref_tokens += m["ref_tokens"]
    if total_ref_tokens == 0:
        return None
    return total_edits / total_ref_tokens
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_app.py::TestMatchRefs tests/test_app.py::TestAggregateWer -v`
Expected: all 6 tests PASS

**Step 5: Commit**

```bash
git add streamlit_app.py tests/test_app.py
git commit -m "Add batch helpers: _match_refs and _aggregate_wer"
```

---

### Task 5: Batch evaluation tab

**Files:**
- Modify: `streamlit_app.py` (add `batch_tab`, update tabs at bottom)
- Modify: `tests/test_app.py` (add `TestBatchTab`, update imports)

**Step 1: Write failing tests**

Update imports in `tests/test_app.py`:

```python
from streamlit_app import (
    transcribe, _wer_label, audio_tab, show_results,
    _match_refs, _aggregate_wer, batch_tab,
)
```

Add test class:

```python
class TestBatchTab:
    @patch("streamlit_app.transcribe")
    @patch("streamlit_app.load_model")
    @patch("streamlit_app.st")
    def test_transcribes_all_files(self, mock_st, mock_load_model, mock_transcribe):
        mock_load_model.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_transcribe.side_effect = ["text one", "text two"]

        audio1 = MagicMock(name="file1.wav")
        audio1.name = "file1.wav"
        audio1.getvalue.return_value = b"audio1"
        audio2 = MagicMock(name="file2.wav")
        audio2.name = "file2.wav"
        audio2.getvalue.return_value = b"audio2"

        mock_st.file_uploader.side_effect = [[audio1, audio2], []]
        mock_st.button.return_value = True
        mock_st.status.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_st.status.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.session_state = {}

        batch_tab()

        assert mock_transcribe.call_count == 2
        results = mock_st.session_state["batch_results"]
        assert len(results) == 2
        assert results[0]["filename"] == "file1.wav"
        assert results[0]["transcription"] == "text one"

    @patch("streamlit_app.st")
    def test_no_results_without_click(self, mock_st):
        mock_st.file_uploader.side_effect = [[], []]
        mock_st.button.return_value = False
        mock_st.session_state = {}

        batch_tab()

        mock_st.dataframe.assert_not_called()
        mock_st.download_button.assert_not_called()

    @patch("streamlit_app.st")
    def test_displays_stored_results(self, mock_st):
        mock_st.file_uploader.side_effect = [[], []]
        mock_st.button.return_value = False
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.session_state = {
            "batch_results": [
                {"filename": "test.wav", "transcription": "hello"},
            ]
        }

        batch_tab()

        mock_st.dataframe.assert_called_once()
        mock_st.download_button.assert_called_once()

    @patch("streamlit_app.transcribe")
    @patch("streamlit_app.load_model")
    @patch("streamlit_app.st")
    def test_computes_wer_when_ref_matched(self, mock_st, mock_load_model, mock_transcribe):
        mock_load_model.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_transcribe.return_value = "hello world"

        audio = MagicMock()
        audio.name = "clip.wav"
        audio.getvalue.return_value = b"audio"
        ref = MagicMock()
        ref.name = "clip.txt"
        ref.getvalue.return_value = b"hello world"

        mock_st.file_uploader.side_effect = [[audio], [ref]]
        mock_st.button.return_value = True
        mock_st.status.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_st.status.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.session_state = {}

        batch_tab()

        results = mock_st.session_state["batch_results"]
        assert "metrics" in results[0]
        assert results[0]["metrics"]["wer"] == 0.0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_app.py::TestBatchTab -v`
Expected: ImportError — `batch_tab` not defined

**Step 3: Implement batch_tab**

Add to `streamlit_app.py` after `_aggregate_wer`:

```python
def batch_tab():
    audio_files = st.file_uploader(
        "Upload audio files",
        type=["wav", "mp3", "flac", "ogg", "webm", "m4a"],
        accept_multiple_files=True,
        key="batch_audio",
    )
    ref_files = st.file_uploader(
        "Upload reference transcripts",
        type=["txt"],
        accept_multiple_files=True,
        key="batch_ref",
        help="Match audio files by filename: patient1.wav \u2194 patient1.txt",
    )

    if st.button("Transcribe All", key="batch_transcribe", disabled=not audio_files):
        ref_map, ref_errors = _match_refs(ref_files)
        for name in ref_errors:
            st.warning(f"Skipping {name}: not a UTF-8 text file.")

        with st.status("Transcribing...", expanded=True) as status:
            st.write("Loading model...")
            processor, model, decoder = load_model()
            results = []
            progress = st.progress(0)
            for i, audio_file in enumerate(audio_files):
                st.write(f"Transcribing {audio_file.name}...")
                text = transcribe(audio_file.getvalue(), processor, model, decoder)
                stem = Path(audio_file.name).stem
                ref = ref_map.get(stem, "")

                entry = {"filename": audio_file.name, "transcription": text}
                if ref.strip():
                    metrics = compute_wer(ref, text)
                    entry["reference"] = ref
                    entry["metrics"] = metrics
                results.append(entry)
                progress.progress((i + 1) / len(audio_files))
            status.update(label="Complete!", state="complete", expanded=False)

        st.session_state["batch_results"] = results
        st.toast("Batch transcription complete!")

    if "batch_results" in st.session_state:
        results = st.session_state["batch_results"]

        st.subheader("Results")
        rows = []
        for r in results:
            m = r.get("metrics")
            rows.append({
                "Filename": r["filename"],
                "WER": f"{m['wer'] * 100:.2f}%" if m else "N/A",
                "Insertions": m["insertions"] if m else "",
                "Deletions": m["deletions"] if m else "",
                "Substitutions": m["substitutions"] if m else "",
            })

        agg = _aggregate_wer(results)
        if agg is not None:
            total_ins = sum(r["metrics"]["insertions"] for r in results if "metrics" in r)
            total_del = sum(r["metrics"]["deletions"] for r in results if "metrics" in r)
            total_sub = sum(r["metrics"]["substitutions"] for r in results if "metrics" in r)
            rows.append({
                "Filename": "AGGREGATE",
                "WER": f"{agg * 100:.2f}%",
                "Insertions": total_ins,
                "Deletions": total_del,
                "Substitutions": total_sub,
            })

        st.dataframe(rows, hide_index=True)

        for r in results:
            with st.expander(r["filename"]):
                st.code(r["transcription"], language=None)
                if "reference" in r:
                    st.markdown(
                        html_diff(r["reference"], r["transcription"]),
                        unsafe_allow_html=True,
                    )

        st.download_button(
            "Download All JSON",
            json.dumps(results, indent=2),
            "batch_results.json",
            "application/json",
            key="batch_dl",
        )
```

**Step 4: Add third tab**

Replace the tabs section at the bottom of `streamlit_app.py`:

```python
tab_upload, tab_record, tab_batch = st.tabs(
    ["Upload Audio", "Record Audio", "Batch Evaluate"]
)
with tab_upload:
    audio_tab(
        st.file_uploader(
            "Upload audio file", type=["wav", "mp3", "flac", "ogg", "webm", "m4a"]
        ),
        "upload",
    )
with tab_record:
    audio_tab(st.audio_input("Record audio"), "record")
with tab_batch:
    batch_tab()
```

**Step 5: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: all tests PASS

**Step 6: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add streamlit_app.py tests/test_app.py
git commit -m "Add batch evaluation tab with multi-file transcription and aggregate WER"
```

---

### Task 6: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update architecture description**

Add to the `streamlit_app.py` bullet in the Architecture section:
- Mention `samples/` directory
- Mention `batch_tab`, `_match_refs`, `_aggregate_wer`
- Mention `st.code` for copy-to-clipboard and corrected transcript

Add `samples/` directory to the file listing.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "Update documentation for UI improvements"
```
