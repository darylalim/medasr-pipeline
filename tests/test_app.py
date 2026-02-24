from unittest.mock import MagicMock, patch

import numpy as np
import torch

from streamlit_app import transcribe, _wer_label, audio_tab, show_results


class TestWerLabel:
    def test_zero_wer(self):
        assert _wer_label(0.0) == "Excellent"

    def test_excellent(self):
        assert _wer_label(0.04) == "Excellent"

    def test_good(self):
        assert _wer_label(0.05) == "Good"

    def test_fair(self):
        assert _wer_label(0.15) == "Fair"

    def test_poor(self):
        assert _wer_label(0.30) == "Poor"

    def test_full_error(self):
        assert _wer_label(1.0) == "Poor"


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


class TestAudioTabFileUpload:
    def _setup_st(self, mock_st, ref_file, text_area_value, key="upload"):
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.file_uploader.return_value = ref_file
        mock_st.text_area.return_value = text_area_value
        mock_st.button.return_value = False
        audio_data = MagicMock()
        audio_data.size = 100
        mock_st.session_state = {
            f"audio_id_{key}": 100,
            f"text_{key}": "transcribed text",
        }
        return audio_data

    @patch("streamlit_app.show_results")
    @patch("streamlit_app.st")
    def test_file_upload_overrides_text_area(self, mock_st, mock_show_results):
        ref_file = MagicMock()
        ref_file.getvalue.return_value = b"reference from file"
        audio_data = self._setup_st(mock_st, ref_file, "typed text")

        audio_tab(audio_data, "upload")

        mock_show_results.assert_called_once_with(
            "transcribed text", "reference from file", "upload"
        )

    @patch("streamlit_app.show_results")
    @patch("streamlit_app.st")
    def test_text_area_used_when_no_file(self, mock_st, mock_show_results):
        audio_data = self._setup_st(mock_st, None, "typed reference")

        audio_tab(audio_data, "upload")

        mock_show_results.assert_called_once_with(
            "transcribed text", "typed reference", "upload"
        )

    @patch("streamlit_app.show_results")
    @patch("streamlit_app.st")
    def test_empty_file_treated_as_no_reference(self, mock_st, mock_show_results):
        ref_file = MagicMock()
        ref_file.getvalue.return_value = b""
        audio_data = self._setup_st(mock_st, ref_file, "")

        audio_tab(audio_data, "upload")

        mock_show_results.assert_called_once_with("transcribed text", "", "upload")

    @patch("streamlit_app.show_results")
    @patch("streamlit_app.st")
    def test_non_utf8_file_shows_error(self, mock_st, mock_show_results):
        ref_file = MagicMock()
        ref_file.getvalue.return_value = b"\xff\xfe"
        audio_data = self._setup_st(mock_st, ref_file, "")

        audio_tab(audio_data, "upload")

        mock_st.error.assert_called_once()
        mock_show_results.assert_called_once_with("transcribed text", "", "upload")


class TestShowResults:
    @patch("streamlit_app.st")
    def test_uses_two_column_layout(self, mock_st):
        col_wer = MagicMock()
        col_breakdown = MagicMock()
        mock_st.columns.return_value = [col_wer, col_breakdown]

        show_results("hello world", "hello world", "test")

        mock_st.columns.assert_called_once_with(2)

    @patch("streamlit_app.st")
    def test_wer_metric_in_left_column(self, mock_st):
        col_wer = MagicMock()
        col_breakdown = MagicMock()
        mock_st.columns.return_value = [col_wer, col_breakdown]

        show_results("hello world", "hello world", "test")

        col_wer.metric.assert_called_once()
        args, kwargs = col_wer.metric.call_args
        assert args[0] == "WER"
        assert args[1] == "0.00%"
        assert kwargs["delta"] == "Excellent"

    @patch("streamlit_app.st")
    def test_dataframe_in_right_column(self, mock_st):
        col_wer = MagicMock()
        col_breakdown = MagicMock()
        mock_st.columns.return_value = [col_wer, col_breakdown]

        show_results("the cat sat", "the dog sat", "test")

        col_breakdown.dataframe.assert_called_once()
        data = col_breakdown.dataframe.call_args[0][0]
        assert data["Type"] == ["Insertions", "Deletions", "Substitutions"]
        assert data["Count"] == [0, 0, 1]
        kwargs = col_breakdown.dataframe.call_args[1]
        assert kwargs["width"] == "stretch"

    @patch("streamlit_app.st")
    def test_no_metrics_without_reference(self, mock_st):
        show_results("hello world", "", "test")

        mock_st.columns.assert_not_called()
