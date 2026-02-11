


"""Auto-generated test stubs for trainable.engines.numpy_engine."""

import pytest
from unittest.mock import MagicMock, patch
import trainable.engines.numpy_engine


class TestNumpyEngine:
    """Tests for NumpyEngine."""

    def setup_method(self):
        """Create a fresh instance for each test."""
        # -- Setup Constructor Arguments --
        config = None
        self.instance = trainable.engines.numpy_engine.NumpyEngine(config)

    def test_spectral_loss(self):
        """Test for NumpyEngine.spectral_loss."""
        # -- Setup --
        synthesized = 0.0
        target = 0.0
        sample_rate = 0
        n_fft = 0
        hop_length = 0
        high_freq_emphasis = 0.0
        # mock_stft = MagicMock(return_value=None)
        # mock_linspace = MagicMock(return_value=None)
        # mock_mean = MagicMock(return_value=None)
        # mock_log = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.spectral_loss(synthesized, target, sample_rate, n_fft, hop_length, high_freq_emphasis)
        # -- Assert --
        assert result == 0.0

    def test_compute_loss(self):
        """Test for NumpyEngine.compute_loss."""
        # -- Setup --
        target_audio = 0.0
        blueprint = None
        sample_rate = 0
        # mock_astype = MagicMock(return_value=None)
        # mock_ParameterContext = MagicMock(return_value=None)
        # mock_arange = MagicMock(return_value=None)
        # mock_getitem_np = MagicMock(return_value=None)
        # mock_spectral_loss = MagicMock(return_value=0.0)
        # mock_mean = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.compute_loss(target_audio, blueprint, sample_rate)
        # -- Assert --
        assert result == 0.0

    def test_step(self):
        """Test for NumpyEngine.step."""
        # -- Setup --
        # mock_backward = MagicMock(return_value=None)
        # mock_zeros = MagicMock(return_value=None)
        # mock_sqrt = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.step()
        # -- Assert --
        assert result == 0.0

    def test_get_parameter_values(self):
        """Test for NumpyEngine.get_parameter_values."""
        # -- Setup --
        # -- Act --
        result = self.instance.get_parameter_values()
        # -- Assert --
        assert result == 0.0

    def test_set_parameter_values(self):
        """Test for NumpyEngine.set_parameter_values."""
        # -- Setup --
        parameters = 0.0
        # -- Act --
        result = self.instance.set_parameter_values(parameters)
        # -- Assert --
        assert result == None
