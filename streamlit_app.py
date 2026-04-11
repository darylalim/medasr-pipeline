import io
from functools import cache
import warnings

for _msg in (
    "Using padding='same' with even kernel lengths",
    "Unigrams not provided",
    "No known unigrams provided",
):
    warnings.filterwarnings("ignore", message=_msg)

from dotenv import load_dotenv  # noqa: E402
import huggingface_hub  # noqa: E402
import librosa  # noqa: E402
import pyctcdecode  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCTC, AutoProcessor  # noqa: E402

load_dotenv()

MODEL_ID = "google/medasr"


@cache
def _detect_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    raise RuntimeError("No GPU found — MedASR requires CUDA or MPS")


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
    device, dtype = _detect_device()
    model = AutoModelForCTC.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
    if device == "cuda":
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


@torch.inference_mode()
def transcribe(audio_bytes: bytes, processor, model, decoder) -> str:
    speech, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    device, dtype = _detect_device()
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt").to(
        device=device, dtype=dtype
    )
    logits = model(**inputs).logits
    log_probs = logits.log_softmax(dim=-1).cpu().float().numpy()[0]
    return (
        decoder.decode_beams(log_probs, beam_width=8)[0][0]
        .translate(_DECODE_TRANS)
        .replace("</s>", "")
        .strip()
    )


def audio_tab(audio_data, key: str) -> None:
    text_key = f"text_{key}"

    if audio_data is not None:
        st.audio(audio_data)

    if st.button(
        "Transcribe",
        key=f"transcribe_{key}",
        disabled=(audio_data is None),
        type="primary",
    ):
        with st.spinner("Transcribing..."):
            text = transcribe(audio_data.getvalue(), *load_model())
        st.session_state[text_key] = text
        st.toast("Transcription complete!")

    if text_key in st.session_state:
        st.text_area(
            "Transcription",
            value=st.session_state[text_key],
            height=300,
            disabled=True,
            label_visibility="collapsed",
        )
        st.download_button(
            "Download",
            data=st.session_state[text_key],
            file_name="transcription.txt",
            key=f"download_{key}",
        )


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
