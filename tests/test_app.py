from unittest.mock import MagicMock, patch

import numpy as np
import torch

from streamlit_app import transcribe


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
