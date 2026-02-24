import io
import json
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

from utils.helper import compute_wer, html_diff  # noqa: E402

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


_CLEANUP = str.maketrans({" ": "", "#": " "})


def transcribe(audio_bytes: bytes, processor, model, decoder) -> str:
    speech, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.to(device=DEVICE, dtype=DTYPE)
    with torch.inference_mode():
        logits = model(**inputs).logits
    log_probs = logits.log_softmax(dim=-1).cpu().float().numpy()[0]
    text = decoder.decode_beams(log_probs, beam_width=8)[0][0]
    return text.translate(_CLEANUP).replace("</s>", "").strip()


def _wer_label(wer: float) -> str:
    if wer < 0.05:
        return "Excellent"
    if wer < 0.15:
        return "Good"
    if wer < 0.30:
        return "Fair"
    return "Poor"


def show_results(text: str, ref_text: str, key: str):
    st.subheader("Transcription")
    st.text_area("Result", value=text, height=200, key=f"result_{key}")

    result = {"transcription": text}
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


def audio_tab(audio_data, key: str):
    with st.expander("Compare against reference transcript"):
        ref = st.text_area(
            "Reference transcript (optional)",
            key=f"ref_{key}",
            height=100,
            help="Paste the expected ground truth text to compute word error rate (WER) metrics against the transcription.",
        )

    if audio_data is not None:
        st.audio(audio_data)

    # Clear stale results when audio changes
    audio_id = audio_data.size if audio_data is not None else None
    if st.session_state.get(f"audio_id_{key}") != audio_id:
        st.session_state[f"audio_id_{key}"] = audio_id
        st.session_state.pop(f"text_{key}", None)

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


st.set_page_config(page_title="MedASR", page_icon="\U0001fa7a", layout="centered")
st.title("Medical Dictation Transcription")

tab_upload, tab_record = st.tabs(["Upload Audio", "Record Audio"])
with tab_upload:
    audio_tab(
        st.file_uploader(
            "Upload audio file", type=["wav", "mp3", "flac", "ogg", "webm", "m4a"]
        ),
        "upload",
    )
with tab_record:
    audio_tab(st.audio_input("Record audio"), "record")
