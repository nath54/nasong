


"""Auto-generated test stubs for trainable.note_detection.torchcrepe_detect."""

import pytest
from unittest.mock import MagicMock, patch
import trainable.note_detection.torchcrepe_detect


class TestTorchCrepeDetector:
    """Tests for TorchCrepeDetector."""

    def setup_method(self):
        """Create a fresh instance for each test."""
        self.instance = trainable.note_detection.torchcrepe_detect.TorchCrepeDetector()

    def test_detect(self):
        """Test for TorchCrepeDetector.detect."""
        # -- Setup --
        audio_segment = None
        sample_rate = 0
        # mock_unsqueeze = MagicMock(return_value=None)
        # mock_get = MagicMock(return_value=None)
        # mock_numpy = MagicMock(return_value=None)
        # mock_ImportError = MagicMock(return_value=None)
        # mock_is_available = MagicMock(return_value=None)
        # mock_predict = MagicMock(return_value=None)
        # mock__add_note = MagicMock(return_value=None)
        # mock_tensor = MagicMock(return_value=None)
        # mock_RuntimeError = MagicMock(return_value=None)
        # mock_cpu = MagicMock(return_value=None)
        # mock_append = MagicMock(return_value=None)
        # mock_squeeze = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.detect(audio_segment, sample_rate)
        # -- Assert --
        assert result == []

    def test__add_note(self):
        """Test for TorchCrepeDetector._add_note."""
        # -- Setup --
        notes = None
        start_idx = None
        end_idx = None
        pitches = None
        confs = None
        hop_s = None
        audio_segment = None
        sample_rate = None
        # mock_median = MagicMock(return_value=None)
        # mock_mean = MagicMock(return_value=None)
        # mock_append = MagicMock(return_value=None)
        # mock_sqrt = MagicMock(return_value=None)
        # -- Act --
        result = self.instance._add_note(notes, start_idx, end_idx, pitches, confs, hop_s, audio_segment, sample_rate)
        # -- Assert --
        assert result == None
