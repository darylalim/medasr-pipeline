# UI Improvements Design

## Overview

Four high-value UI improvements to the MedASR Streamlit app, aimed at making the developer experience smoother and supporting medical dictation evaluation workflows.

## 1. Sample Audio

- Add a `samples/` directory with a demo audio file and reference transcript (copied from test data, kept independent)
- In both Upload and Record tabs, add a "Try with sample" button above the file uploader
- Clicking it loads the sample audio and reference transcript into session state, runs transcription, and shows results

## 2. Copy-to-Clipboard

- Replace `st.text_area("Result", ...)` with `st.code(text)` for built-in copy button
- Transcription output becomes read-only with native copy functionality

## 3. Corrected Transcript

- Below the `st.code` transcription output, add a `st.text_area("Corrected transcript", value=text)` pre-filled with the original transcription
- User can freely edit the corrected field
- JSON download includes both `transcription` (original) and `corrected_transcription` (edited, only if changed)
- WER metrics stay anchored to the original transcription

## 4. Batch Evaluation

- Third tab labeled "Batch Evaluate" alongside "Upload Audio" and "Record Audio"
- Two multi-file uploaders: audio files (wav, mp3, flac, ogg, webm, m4a) and reference transcript `.txt` files
- Matching by filename stem: `patient1.wav` matches `patient1.txt`. Audio without a matching transcript is transcribed without WER
- "Transcribe All" button processes files sequentially with a progress bar
- Results displayed as a summary table: filename, WER, insertions, deletions, substitutions
- Aggregate WER row at the bottom
- Expandable rows for individual transcription text and word diff
- Single "Download All JSON" button exports an array of all results
