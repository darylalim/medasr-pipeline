import io
import json
import warnings

warnings.filterwarnings(
    "ignore", message="Using padding='same' with even kernel lengths"
)
warnings.filterwarnings("ignore", message="Unigrams not provided")
warnings.filterwarnings("ignore", message="No known unigrams provided")

from dotenv import load_dotenv  # noqa: E402
import huggingface_hub  # noqa: E402
import librosa  # noqa: E402
import pyctcdecode  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCTC, AutoProcessor  # noqa: E402

from utils.helper import compute_wer  # noqa: E402

load_dotenv()

MODEL_ID = "google/medasr"
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCTC.from_pretrained(MODEL_ID).to(DEVICE).eval()
    lm_path = huggingface_hub.hf_hub_download(MODEL_ID, filename="lm_6.kenlm")
    tokenizer = processor.tokenizer
    vocab = [""] * tokenizer.vocab_size
    for token, idx in tokenizer.vocab.items():
        if idx < tokenizer.vocab_size and idx > 0:
            if token.startswith("<") and token.endswith(">"):
                vocab[idx] = token
            else:
                vocab[idx] = "▁" + token.replace("▁", "#")
    decoder = pyctcdecode.build_ctcdecoder(vocab, lm_path)
    return processor, model, decoder


def transcribe(audio_bytes: bytes, processor, model, decoder) -> str:
    speech, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    inputs = processor(speech, sampling_rate=16000).to(DEVICE)
    with torch.inference_mode():
        logits = model(**inputs).logits
    log_probs = logits.log_softmax(dim=-1).cpu().numpy()[0]
    text = decoder.decode_beams(log_probs, beam_width=8)[0][0]
    return text.replace(" ", "").replace("#", " ").replace("</s>", "").strip()


def show_results(text: str, ref_text: str, key: str):
    st.subheader("Transcription")
    st.text_area("Result", value=text, height=200, key=f"result_{key}")

    result = {"transcription": text}
    if ref_text.strip():
        metrics = compute_wer(ref_text, text)
        result["reference"] = ref_text
        result["metrics"] = metrics
        st.subheader("Evaluation Metrics")
        cols = st.columns(4)
        for col, (label, field) in zip(
            cols,
            [
                ("WER", "wer"),
                ("Insertions", "insertions"),
                ("Deletions", "deletions"),
                ("Substitutions", "substitutions"),
            ],
        ):
            value = f"{metrics[field] * 100:.2f}%" if field == "wer" else metrics[field]
            col.metric(label, value)

    st.download_button(
        "Download JSON",
        json.dumps(result, indent=2),
        "transcription_result.json",
        "application/json",
        key=f"dl_{key}",
    )


def audio_tab(audio_data, key: str):
    ref = st.text_area(
        "Reference transcript (optional)",
        key=f"ref_{key}",
        height=100,
        help="Paste the expected ground truth text to compute word error rate (WER) metrics against the transcription.",
    )
    if audio_data is not None:
        st.audio(audio_data)
        if st.button("Transcribe", key=f"transcribe_{key}"):
            with st.spinner("Transcribing..."):
                text = transcribe(audio_data.getvalue(), *load_model())
            show_results(text, ref, key)


st.title("Medical Dictation Transcription")
st.write("Transcribe medical dictation using Google's MedASR model.")

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
