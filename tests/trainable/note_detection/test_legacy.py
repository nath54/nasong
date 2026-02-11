


"""Auto-generated test stubs for trainable.note_detection.legacy."""

import pytest
from unittest.mock import MagicMock, patch
import trainable.note_detection.legacy


class TestLegacyDetector:
    """Tests for LegacyDetector."""

    def setup_method(self):
        """Create a fresh instance for each test."""
        self.instance = trainable.note_detection.legacy.LegacyDetector()

    def test_detect(self):
        """Test for LegacyDetector.detect."""
        # -- Setup --
        audio_segment = None
        sample_rate = 0
        # mock_get = MagicMock(return_value=None)
        # mock_array = MagicMock(return_value=None)
        # mock__detect_pitches_fft = MagicMock(return_value=[])
        # mock_append = MagicMock(return_value=None)
        # mock_sqrt = MagicMock(return_value=None)
        # mock_mean = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.detect(audio_segment, sample_rate)
        # -- Assert --
        assert result == []

    def test__detect_pitches_fft(self):
        """Test for LegacyDetector._detect_pitches_fft."""
        # -- Setup --
        audio_segment = None
        sample_rate = 0
        max_pitches = 0
        min_freq = 0.0
        max_freq = 0.0
        # mock_rfft = MagicMock(return_value=None)
        # mock_rfftfreq = MagicMock(return_value=None)
        # mock_sort = MagicMock(return_value=None)
        # mock_hanning = MagicMock(return_value=None)
        # mock_append = MagicMock(return_value=None)
        # -- Act --
        result = self.instance._detect_pitches_fft(audio_segment, sample_rate, max_pitches, min_freq, max_freq)
        # -- Assert --
        assert result == []
