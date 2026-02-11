


"""Auto-generated test stubs for trainable.instruments.melodic."""

import pytest
from unittest.mock import MagicMock, patch
import trainable.instruments.melodic


def test_TrainablePlucked():
    """Test for TrainablePlucked."""
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
    # mock_ExponentialDecay = MagicMock(return_value=None)
    # mock_BandLimitedSawtooth = MagicMock(return_value=None)
    # mock_Product = MagicMock(return_value=None)
    # -- Act --
    result = trainable.instruments.melodic.TrainablePlucked(time, frequency, start_time, duration, init_amplitude, name_prefix)
    # -- Assert --
    assert result == None

def test_TrainablePiano():
    """Test for TrainablePiano."""
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
    # mock_Sin = MagicMock(return_value=None)
    # mock_Sum = MagicMock(return_value=None)
    # -- Act --
    result = trainable.instruments.melodic.TrainablePiano(time, frequency, start_time, duration, init_amplitude, name_prefix)
    # -- Assert --
    assert result == None

def test_TrainableBowed():
    """Test for TrainableBowed."""
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
    # mock_Sin = MagicMock(return_value=None)
    # mock_Product = MagicMock(return_value=None)
    # mock_BandLimitedSawtooth = MagicMock(return_value=None)
    # mock_Sum = MagicMock(return_value=None)
    # -- Act --
    result = trainable.instruments.melodic.TrainableBowed(time, frequency, start_time, duration, init_amplitude, name_prefix)
    # -- Assert --
    assert result == None
