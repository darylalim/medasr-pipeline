import io
import json
import warnings
from pathlib import Path

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

from utils.helper import compute_wer, html_diff, normalize  # noqa: E402

load_dotenv()

MODEL_ID = "google/medasr"
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SAMPLE_DIR = Path(__file__).parent / "samples"
SAMPLE_AUDIO = SAMPLE_DIR / "sample_audio.wav"
SAMPLE_REF = SAMPLE_DIR / "sample_transcript.txt"


def _patch_feature_extractor(feature_extractor):
    """Work around transformers 5.2.0 bug that passes an extra `center` arg
    to _torch_extract_fbank_features (huggingface/transformers#38341)."""
    original = feature_extractor._torch_extract_fbank_features

    def _patched(waveform, device="cpu", center=None):
        return original(waveform, device)

    feature_extractor._torch_extract_fbank_features = _patched


@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    _patch_feature_extractor(processor.feature_extractor)
    model = AutoModelForCTC.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    if DEVICE == "cuda":
        model = torch.compile(model)
    lm_path = huggingface_hub.hf_hub_download(MODEL_ID, filename="lm_6.kenlm")
    tokenizer = processor.tokenizer
    vocab = [""] * tokenizer.vocab_size
    for token, idx in tokenizer.vocab.items():
        if 0 < idx < tokenizer.vocab_size:
            vocab[idx] = (
                token
                if token.startswith("<") and token.endswith(">")
                else "▁" + token.replace("▁", "#")
            )
    decoder = pyctcdecode.build_ctcdecoder(vocab, lm_path)
    return processor, model, decoder


# CTC decoder outputs '#' as word separator and literal spaces as noise
_DECODE_TRANS = str.maketrans({" ": "", "#": " "})


def transcribe(audio_bytes: bytes, processor, model, decoder) -> str:
    speech, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.to(device=DEVICE, dtype=DTYPE)
    with torch.inference_mode():
        logits = model(**inputs).logits
    log_probs = logits.log_softmax(dim=-1).cpu().float().numpy()[0]
    text = decoder.decode_beams(log_probs, beam_width=8)[0][0]
    return text.translate(_DECODE_TRANS).replace("</s>", "").strip()


def _wer_label(wer: float) -> str:
    if wer < 0.05:
        return "Excellent"
    if wer < 0.15:
        return "Good"
    if wer < 0.30:
        return "Fair"
    return "Poor"


def show_results(text: str, ref_text: str, key: str) -> None:
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

    if normalize(ref_text):
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
            use_container_width=True,
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


def audio_tab(audio_data, key: str) -> None:
    if SAMPLE_AUDIO.exists():
        if st.button("Try with sample", key=f"sample_{key}"):
            sample_bytes = SAMPLE_AUDIO.read_bytes()
            with st.status("Transcribing sample...", expanded=True) as status:
                st.write("Loading model...")
                processor, model, decoder = load_model()
                st.write("Transcribing audio...")
                text = transcribe(sample_bytes, processor, model, decoder)
                status.update(label="Complete!", state="complete", expanded=False)
            st.session_state[f"text_{key}"] = text
            st.session_state[f"audio_id_{key}"] = "sample"
            st.session_state[f"sample_bytes_{key}"] = sample_bytes
            if SAMPLE_REF.exists():
                st.session_state[f"sample_ref_{key}"] = SAMPLE_REF.read_text(
                    encoding="utf-8"
                )
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
        st.audio(st.session_state.get(f"sample_bytes_{key}", b""), format="audio/wav")

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
        st.session_state.pop(f"sample_bytes_{key}", None)

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


def _match_refs(ref_files: list) -> tuple[dict[str, str], list[str]]:
    ref_map = {}
    errors = []
    for f in ref_files:
        stem = Path(f.name).stem
        try:
            ref_map[stem] = f.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f.name)
    return ref_map, errors


def _aggregate_wer(results: list[dict]) -> float | None:
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


def batch_tab() -> None:
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
                if normalize(ref):
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
            rows.append(
                {
                    "Filename": r["filename"],
                    "WER": f"{m['wer'] * 100:.2f}%" if m else "N/A",
                    "Insertions": m["insertions"] if m else "",
                    "Deletions": m["deletions"] if m else "",
                    "Substitutions": m["substitutions"] if m else "",
                }
            )

        agg = _aggregate_wer(results)
        if agg is not None:
            total_ins = sum(
                r["metrics"]["insertions"] for r in results if "metrics" in r
            )
            total_del = sum(
                r["metrics"]["deletions"] for r in results if "metrics" in r
            )
            total_sub = sum(
                r["metrics"]["substitutions"] for r in results if "metrics" in r
            )
            rows.append(
                {
                    "Filename": "AGGREGATE",
                    "WER": f"{agg * 100:.2f}%",
                    "Insertions": total_ins,
                    "Deletions": total_del,
                    "Substitutions": total_sub,
                }
            )

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


st.set_page_config(page_title="MedASR", page_icon="\U0001fa7a", layout="centered")
st.title("Medical Dictation Transcription")

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
