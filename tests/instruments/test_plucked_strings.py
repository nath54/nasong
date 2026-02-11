


"""Auto-generated test stubs for instruments.plucked_strings."""

import pytest
from unittest.mock import MagicMock, patch
import instruments.plucked_strings


def test_GuitarString():
    """Test for GuitarString."""
    # -- Setup --
    time = None
    frequency = 0.0
    start_time = 0.0
    duration = 0.0
    brightness = 0.0
    # mock_ExponentialADSR = MagicMock(return_value=None)
    # mock_BasicScaling = MagicMock(return_value=None)
    # mock_Sin = MagicMock(return_value=None)
    # mock_ExponentialDecay = MagicMock(return_value=None)
    # mock_Sum = MagicMock(return_value=None)
    # mock_Product = MagicMock(return_value=None)
    # mock_c = MagicMock(return_value=None)
    # -- Act --
    result = instruments.plucked_strings.GuitarString(time, frequency, start_time, duration, brightness)
    # -- Assert --
    assert result == None

def test_GuitarString2():
    """Test for GuitarString2."""
    # -- Setup --
    time = None
    frequency = 0.0
    start_time = 0.0
    duration = 0.0
    amplitude = 0.0
    # mock_ExponentialADSR = MagicMock(return_value=None)
    # mock_Sin = MagicMock(return_value=None)
    # mock_WhiteNoise = MagicMock(return_value=None)
    # mock_Sum = MagicMock(return_value=None)
    # mock_Product = MagicMock(return_value=None)
    # mock_c = MagicMock(return_value=None)
    # -- Act --
    result = instruments.plucked_strings.GuitarString2(time, frequency, start_time, duration, amplitude)
    # -- Assert --
    assert result == None

def test_AcousticString():
    """Test for AcousticString."""
    # -- Setup --
    time = None
    frequency = 0.0
    pluck_time = 0.0
    amplitude = 0.0
    decay_rate = 0.0
    # mock_ADSR2 = MagicMock(return_value=None)
    # mock_ExponentialDecay = MagicMock(return_value=None)
    # mock_Product = MagicMock(return_value=None)
    # mock_Sin = MagicMock(return_value=None)
    # mock_WhiteNoise = MagicMock(return_value=None)
    # mock_Sum = MagicMock(return_value=None)
    # mock_c = MagicMock(return_value=None)
    # -- Act --
    result = instruments.plucked_strings.AcousticString(time, frequency, pluck_time, amplitude, decay_rate)
    # -- Assert --
    assert result == None

def test_Fingerpicking():
    """Test for Fingerpicking."""
    # -- Setup --
    time = None
    bass_note = 0.0
    chord_notes = []
    start_time = 0.0
    pattern_duration = 0.0
    # mock_append = MagicMock(return_value=None)
    # mock_Sequencer = MagicMock(return_value=None)
    # mock_AcousticString = MagicMock(return_value=None)
    # -- Act --
    result = instruments.plucked_strings.Fingerpicking(time, bass_note, chord_notes, start_time, pattern_duration)
    # -- Assert --
    assert result == None

def test_Strum():
    """Test for Strum."""
    # -- Setup --
    time = None
    frequencies = []
    start_time = 0.0
    duration = 0.0
    # mock_Sequencer = MagicMock(return_value=None)
    # mock_append = MagicMock(return_value=None)
    # mock_GuitarString = MagicMock(return_value=None)
    # -- Act --
    result = instruments.plucked_strings.Strum(time, frequencies, start_time, duration)
    # -- Assert --
    assert result == None
