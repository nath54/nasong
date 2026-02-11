


"""Auto-generated test stubs for trainable.note_detection.onnx_crepe_detect."""

import pytest
from unittest.mock import MagicMock, patch
import trainable.note_detection.onnx_crepe_detect


class TestOnnxCrepeDetector:
    """Tests for OnnxCrepeDetector."""

    def setup_method(self):
        """Create a fresh instance for each test."""
        self.instance = trainable.note_detection.onnx_crepe_detect.OnnxCrepeDetector()

    def test_detect(self):
        """Test for OnnxCrepeDetector.detect."""
        # -- Setup --
        audio_data = None
        sample_rate = 0
        # mock_join = MagicMock(return_value=None)
        # mock_makedirs = MagicMock(return_value=None)
        # mock_pad = MagicMock(return_value=None)
        # mock_array = MagicMock(return_value=None)
        # mock_InferenceSession = MagicMock(return_value=None)
        # mock_concatenate = MagicMock(return_value=None)
        # mock_get = MagicMock(return_value=None)
        # mock_ImportError = MagicMock(return_value=None)
        # mock_dirname = MagicMock(return_value=None)
        # mock_exists = MagicMock(return_value=None)
        # mock_run = MagicMock(return_value=None)
        # mock_append = MagicMock(return_value=None)
        # mock_argmax = MagicMock(return_value=None)
        # mock_arange = MagicMock(return_value=None)
        # mock__finish_note = MagicMock(return_value=None)
        # mock_urlretrieve = MagicMock(return_value=None)
        # mock_resample = MagicMock(return_value=None)
        # mock_linspace = MagicMock(return_value=None)
        # mock_interp = MagicMock(return_value=None)
        # mock_mean = MagicMock(return_value=None)
        # mock_std = MagicMock(return_value=None)
        # mock_frame_generator = MagicMock(return_value=None)
        # mock_get_inputs = MagicMock(return_value=None)
        # mock_RuntimeError = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.detect(audio_data, sample_rate)
        # -- Assert --
        assert result == ""

    def test__finish_note(self):
        """Test for OnnxCrepeDetector._finish_note."""
        # -- Setup --
        notes_list = None
        note_data = None
        audio_data = None
        sample_rate = 44100
        # mock_median = MagicMock(return_value=None)
        # mock_mean = MagicMock(return_value=None)
        # mock_append = MagicMock(return_value=None)
        # mock_sqrt = MagicMock(return_value=None)
        # -- Act --
        result = self.instance._finish_note(notes_list, note_data, audio_data, sample_rate)
        # -- Assert --
        assert result == None
