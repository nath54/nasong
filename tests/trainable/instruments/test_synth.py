


"""Auto-generated test stubs for trainable.instruments.synth."""

import pytest
from unittest.mock import MagicMock, patch
import trainable.instruments.synth


def test_TrainableSawtoothSynth():
    """Test for TrainableSawtoothSynth."""
    # -- Setup --
    time = None
    frequency = None
    start_time = 0.0
    duration = 0.0
    init_amplitude = 0.0
    name_prefix = ""
    # mock_Constant = MagicMock(return_value=None)
    # mock_ValueTrainableParameter = MagicMock(return_value=None)
    # mock_BandLimitedSawtooth = MagicMock(return_value=None)
    # mock_ExponentialADSR = MagicMock(return_value=None)
    # mock_Product = MagicMock(return_value=None)
    # -- Act --
    result = trainable.instruments.synth.TrainableSawtoothSynth(time, frequency, start_time, duration, init_amplitude, name_prefix)
    # -- Assert --
    assert result == None

def test_TrainableSquareSynth():
    """Test for TrainableSquareSynth."""
    # -- Setup --
    time = None
    frequency = None
    start_time = 0.0
    duration = 0.0
    init_amplitude = 0.0
    name_prefix = ""
    # mock_Constant = MagicMock(return_value=None)
    # mock_ValueTrainableParameter = MagicMock(return_value=None)
    # mock_BandLimitedSquare = MagicMock(return_value=None)
    # mock_ExponentialADSR = MagicMock(return_value=None)
    # mock_Product = MagicMock(return_value=None)
    # -- Act --
    result = trainable.instruments.synth.TrainableSquareSynth(time, frequency, start_time, duration, init_amplitude, name_prefix)
    # -- Assert --
    assert result == None

def test_TrainableSineSynth():
    """Test for TrainableSineSynth."""
    # -- Setup --
    time = None
    frequency = None
    start_time = 0.0
    duration = 0.0
    init_amplitude = 0.0
    name_prefix = ""
    # mock_Constant = MagicMock(return_value=None)
    # mock_ValueTrainableParameter = MagicMock(return_value=None)
    # mock_Product = MagicMock(return_value=None)
    # mock_Sin = MagicMock(return_value=None)
    # mock_ExponentialADSR = MagicMock(return_value=None)
    # -- Act --
    result = trainable.instruments.synth.TrainableSineSynth(time, frequency, start_time, duration, init_amplitude, name_prefix)
    # -- Assert --
    assert result == None
