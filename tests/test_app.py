import json
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from streamlit_app import (
    transcribe,
    _wer_label,
    audio_tab,
    show_results,
    _match_refs,
    _aggregate_wer,
    batch_tab,
)


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
    def _setup_st(
        self, mock_st, mock_sample_audio, ref_file, text_area_value, key="upload"
    ):
        mock_sample_audio.exists.return_value = False
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
    @patch("streamlit_app.SAMPLE_AUDIO")
    @patch("streamlit_app.st")
    def test_file_upload_overrides_text_area(
        self, mock_st, mock_sample_audio, mock_show_results
    ):
        ref_file = MagicMock()
        ref_file.getvalue.return_value = b"reference from file"
        audio_data = self._setup_st(mock_st, mock_sample_audio, ref_file, "typed text")

        audio_tab(audio_data, "upload")

        mock_show_results.assert_called_once_with(
            "transcribed text", "reference from file", "upload"
        )

    @patch("streamlit_app.show_results")
    @patch("streamlit_app.SAMPLE_AUDIO")
    @patch("streamlit_app.st")
    def test_text_area_used_when_no_file(
        self, mock_st, mock_sample_audio, mock_show_results
    ):
        audio_data = self._setup_st(mock_st, mock_sample_audio, None, "typed reference")

        audio_tab(audio_data, "upload")

        mock_show_results.assert_called_once_with(
            "transcribed text", "typed reference", "upload"
        )

    @patch("streamlit_app.show_results")
    @patch("streamlit_app.SAMPLE_AUDIO")
    @patch("streamlit_app.st")
    def test_empty_file_treated_as_no_reference(
        self, mock_st, mock_sample_audio, mock_show_results
    ):
        ref_file = MagicMock()
        ref_file.getvalue.return_value = b""
        audio_data = self._setup_st(mock_st, mock_sample_audio, ref_file, "")

        audio_tab(audio_data, "upload")

        mock_show_results.assert_called_once_with("transcribed text", "", "upload")

    @patch("streamlit_app.show_results")
    @patch("streamlit_app.SAMPLE_AUDIO")
    @patch("streamlit_app.st")
    def test_non_utf8_file_shows_error(
        self, mock_st, mock_sample_audio, mock_show_results
    ):
        ref_file = MagicMock()
        ref_file.getvalue.return_value = b"\xff\xfe"
        audio_data = self._setup_st(mock_st, mock_sample_audio, ref_file, "")

        audio_tab(audio_data, "upload")

        mock_st.error.assert_called_once()
        mock_show_results.assert_called_once_with("transcribed text", "", "upload")


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
    @patch(
        "streamlit_app.load_model", return_value=(MagicMock(), MagicMock(), MagicMock())
    )
    @patch("streamlit_app.SAMPLE_REF")
    @patch("streamlit_app.SAMPLE_AUDIO")
    @patch("streamlit_app.st")
    def test_sample_click_transcribes_and_stores(
        self,
        mock_st,
        mock_sample_audio,
        mock_sample_ref,
        mock_load_model,
        mock_transcribe,
        mock_show_results,
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
    def test_sample_ref_used_as_fallback(
        self, mock_st, mock_sample_audio, mock_show_results
    ):
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

        mock_show_results.assert_called_once_with(
            "transcribed", "sample ref text", "upload"
        )


class TestShowResults:
    @patch("streamlit_app.st")
    def test_uses_two_column_layout(self, mock_st):
        mock_st.text_area.return_value = "hello world"
        col_wer = MagicMock()
        col_breakdown = MagicMock()
        mock_st.columns.return_value = [col_wer, col_breakdown]

        show_results("hello world", "hello world", "test")

        mock_st.columns.assert_called_once_with(2)

    @patch("streamlit_app.st")
    def test_wer_metric_in_left_column(self, mock_st):
        mock_st.text_area.return_value = "hello world"
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
        mock_st.text_area.return_value = "the cat sat"
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
        mock_st.text_area.return_value = "hello world"
        show_results("hello world", "", "test")

        mock_st.columns.assert_not_called()

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
            {
                "metrics": {
                    "insertions": 1,
                    "deletions": 0,
                    "substitutions": 0,
                    "ref_tokens": 10,
                    "wer": 0.1,
                }
            },
            {
                "metrics": {
                    "insertions": 0,
                    "deletions": 1,
                    "substitutions": 1,
                    "ref_tokens": 10,
                    "wer": 0.2,
                }
            },
        ]
        assert _aggregate_wer(results) == 3 / 20

    def test_no_metrics_returns_none(self):
        results = [{"filename": "test.wav", "transcription": "hello"}]
        assert _aggregate_wer(results) is None

    def test_mixed_with_and_without_metrics(self):
        results = [
            {
                "metrics": {
                    "insertions": 2,
                    "deletions": 0,
                    "substitutions": 0,
                    "ref_tokens": 10,
                    "wer": 0.2,
                }
            },
            {"filename": "no_ref.wav", "transcription": "hello"},
        ]
        assert _aggregate_wer(results) == 2 / 10


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
    def test_computes_wer_when_ref_matched(
        self, mock_st, mock_load_model, mock_transcribe
    ):
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
