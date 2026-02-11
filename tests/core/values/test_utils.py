


"""Auto-generated test stubs for core.values.utils."""

import pytest
from unittest.mock import MagicMock, patch
import core.values.utils


def test_generate_harmonics():
    """Test for generate_harmonics."""
    # -- Setup --
    time = None
    base_frequency = 0.0
    num_harmonics = 0
    amplitude_falloff = 0.0
    sample_rate = 0
    base_amplitude = None
    # mock_Constant = MagicMock(return_value=None)
    # mock_Sum = MagicMock(return_value=None)
    # mock_Product = MagicMock(return_value=None)
    # mock_append = MagicMock(return_value=None)
    # mock_Sin = MagicMock(return_value=None)
    # -- Act --
    result = core.values.utils.generate_harmonics(time, base_frequency, num_harmonics, amplitude_falloff, sample_rate, base_amplitude)
    # -- Assert --
    assert result == None

def test_LFO():
    """Test for LFO."""
    # -- Setup --
    time = None
    rate_hz = None
    waveform_class = None
    amplitude = None
    delta = None
    # mock_Constant = MagicMock(return_value=None)
    # mock_waveform_class = MagicMock(return_value=None)
    # mock_Product = MagicMock(return_value=None)
    # -- Act --
    result = core.values.utils.LFO(time, rate_hz, waveform_class, amplitude, delta)
    # -- Assert --
    assert result == None
