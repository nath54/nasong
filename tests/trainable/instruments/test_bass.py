


"""Auto-generated test stubs for trainable.instruments.bass."""

import pytest
from unittest.mock import MagicMock, patch
import trainable.instruments.bass


def test_TrainableBass():
    """Test for TrainableBass."""
    # -- Setup --
    time = None
    frequency = None
    start_time = 0.0
    duration = 0.0
    init_amplitude = 0.0
    name_prefix = ""
    # mock_Constant = MagicMock(return_value=None)
    # mock_ValueTrainableParameter = MagicMock(return_value=None)
    # mock_ExponentialADSR = MagicMock(return_value=None)
    # mock_Product = MagicMock(return_value=None)
    # mock_BandLimitedSawtooth = MagicMock(return_value=None)
    # mock_Sin = MagicMock(return_value=None)
    # mock_Sum = MagicMock(return_value=None)
    # mock_Distortion = MagicMock(return_value=None)
    # -- Act --
    result = trainable.instruments.bass.TrainableBass(time, frequency, start_time, duration, init_amplitude, name_prefix)
    # -- Assert --
    assert result == None
