from unittest.mock import MagicMock, patch

import numpy as np
import torch

from streamlit_app import (
    transcribe,
    _patch_feature_extractor,
    audio_tab,
)


class TestTranscribe:
    def _make_mocks(self, decoded_text="hello world"):
        processor = MagicMock()
        inputs = MagicMock()
        inputs.to.return_value = inputs
        processor.return_value = inputs

        model = MagicMock()
        logits = torch.randn(1, 10, 32)
        model_output = MagicMock()
        model_output.logits = logits
        model.return_value = model_output

        decoder = MagicMock()
        decoder.decode_beams.return_value = [(decoded_text, None, None, None)]

        return processor, model, decoder

    @patch("streamlit_app.librosa")
    def test_returns_decoded_text(self, mock_librosa):
        mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        processor, model, decoder = self._make_mocks("hello#world")
        result = transcribe(b"fake_audio", processor, model, decoder)
        assert result == "hello world"

    @patch("streamlit_app.librosa")
    def test_strips_eos_token(self, mock_librosa):
        mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        processor, model, decoder = self._make_mocks("hello</s>")
        result = transcribe(b"fake_audio", processor, model, decoder)
        assert "</s>" not in result

    @patch("streamlit_app.librosa")
    def test_removes_spaces_from_decoder_output(self, mock_librosa):
        mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        processor, model, decoder = self._make_mocks(" hello#world")
        result = transcribe(b"fake_audio", processor, model, decoder)
        assert result == "hello world"

    @patch("streamlit_app.librosa")
    def test_loads_audio_at_16khz(self, mock_librosa):
        mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        processor, model, decoder = self._make_mocks()
        transcribe(b"fake_audio", processor, model, decoder)
        _, kwargs = mock_librosa.load.call_args
        assert kwargs["sr"] == 16000

    @patch("streamlit_app.librosa")
    def test_uses_beam_width_8(self, mock_librosa):
        mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        processor, model, decoder = self._make_mocks()
        transcribe(b"fake_audio", processor, model, decoder)
        _, kwargs = decoder.decode_beams.call_args
        assert kwargs["beam_width"] == 8

    @patch("streamlit_app.librosa")
    def test_passes_return_tensors_pt(self, mock_librosa):
        mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        processor, model, decoder = self._make_mocks()
        transcribe(b"fake_audio", processor, model, decoder)
        _, kwargs = processor.call_args
        assert kwargs["return_tensors"] == "pt"


class TestPatchFeatureExtractor:
    def test_ignores_center_argument(self):
        feature_extractor = MagicMock()
        original_fn = MagicMock(return_value="result")
        feature_extractor._torch_extract_fbank_features = original_fn

        _patch_feature_extractor(feature_extractor)

        result = feature_extractor._torch_extract_fbank_features(
            "waveform", device="cpu", center=True
        )
        original_fn.assert_called_once_with("waveform", "cpu")
        assert result == "result"

    def test_works_without_center(self):
        feature_extractor = MagicMock()
        original_fn = MagicMock(return_value="result")
        feature_extractor._torch_extract_fbank_features = original_fn

        _patch_feature_extractor(feature_extractor)

        result = feature_extractor._torch_extract_fbank_features("waveform")
        original_fn.assert_called_once_with("waveform", "cpu")
        assert result == "result"


class TestAudioTab:
    @patch("streamlit_app.st")
    def test_transcribe_button_disabled_without_audio(self, mock_st):
        mock_st.button.return_value = False
        mock_st.session_state = {}

        audio_tab(None, "upload")

        mock_st.button.assert_called_once_with(
            "Transcribe", key="transcribe_upload", disabled=True
        )

    @patch("streamlit_app.st")
    def test_transcribe_button_enabled_with_audio(self, mock_st):
        mock_st.button.return_value = False
        mock_st.session_state = {}
        audio_data = MagicMock()

        audio_tab(audio_data, "upload")

        mock_st.button.assert_called_once_with(
            "Transcribe", key="transcribe_upload", disabled=False
        )

    @patch("streamlit_app.st")
    def test_plays_audio_when_provided(self, mock_st):
        mock_st.button.return_value = False
        mock_st.session_state = {}
        audio_data = MagicMock()

        audio_tab(audio_data, "upload")

        mock_st.audio.assert_called_once_with(audio_data)

    @patch("streamlit_app.st")
    def test_no_audio_playback_without_data(self, mock_st):
        mock_st.button.return_value = False
        mock_st.session_state = {}

        audio_tab(None, "upload")

        mock_st.audio.assert_not_called()

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

    @patch("streamlit_app.transcribe", return_value="hello world")
    @patch(
        "streamlit_app.load_model",
        return_value=(MagicMock(), MagicMock(), MagicMock()),
    )
    @patch("streamlit_app.st")
    def test_toast_shown_after_transcription(
        self, mock_st, mock_load_model, mock_transcribe
    ):
        mock_st.button.return_value = True
        mock_st.status.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_st.status.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.session_state = {}
        audio_data = MagicMock()
        audio_data.getvalue.return_value = b"fake_audio"

        audio_tab(audio_data, "upload")

        mock_st.toast.assert_called_once_with("Transcription complete!")

    @patch("streamlit_app.st")
    def test_displays_transcription_when_in_session(self, mock_st):
        mock_st.button.return_value = False
        mock_st.session_state = {"text_upload": "transcribed text"}

        audio_tab(None, "upload")

        mock_st.subheader.assert_called_once_with("Transcription")
        mock_st.text_area.assert_called_once_with(
            "Transcription",
            value="transcribed text",
            disabled=True,
            label_visibility="collapsed",
        )
