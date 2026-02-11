


"""Auto-generated test stubs for trainable.instruments.percussion."""

import pytest
from unittest.mock import MagicMock, patch
import trainable.instruments.percussion


def test_TrainableKick():
    """Test for TrainableKick."""
    # -- Setup --
    time = None
    start_time = 0.0
    name_prefix = ""
    # mock_ValueTrainableParameter = MagicMock(return_value=None)
    # mock_ExponentialDecay = MagicMock(return_value=None)
    # mock_Sum = MagicMock(return_value=None)
    # mock_Product = MagicMock(return_value=None)
    # mock_Sin = MagicMock(return_value=None)
    # mock_WhiteNoise = MagicMock(return_value=None)
    # mock_Constant = MagicMock(return_value=None)
    # -- Act --
    result = trainable.instruments.percussion.TrainableKick(time, start_time, name_prefix)
    # -- Assert --
    assert result == None

def test_TrainableSnare():
    """Test for TrainableSnare."""
    # -- Setup --
    time = None
    start_time = 0.0
    name_prefix = ""
    # mock_ValueTrainableParameter = MagicMock(return_value=None)
    # mock_ExponentialDecay = MagicMock(return_value=None)
    # mock_Sin = MagicMock(return_value=None)
    # mock_WhiteNoise = MagicMock(return_value=None)
    # mock_Sum = MagicMock(return_value=None)
    # mock_Product = MagicMock(return_value=None)
    # mock_Constant = MagicMock(return_value=None)
    # -- Act --
    result = trainable.instruments.percussion.TrainableSnare(time, start_time, name_prefix)
    # -- Assert --
    assert result == None

def test_TrainableHiHat():
    """Test for TrainableHiHat."""
    # -- Setup --
    time = None
    start_time = 0.0
    is_open = False
    name_prefix = ""
    # mock_ValueTrainableParameter = MagicMock(return_value=None)
    # mock_ExponentialDecay = MagicMock(return_value=None)
    # mock_WhiteNoise = MagicMock(return_value=None)
    # mock_Sum = MagicMock(return_value=None)
    # mock_Product = MagicMock(return_value=None)
    # mock_Sin = MagicMock(return_value=None)
    # mock_Constant = MagicMock(return_value=None)
    # -- Act --
    result = trainable.instruments.percussion.TrainableHiHat(time, start_time, is_open, name_prefix)
    # -- Assert --
    assert result == None
