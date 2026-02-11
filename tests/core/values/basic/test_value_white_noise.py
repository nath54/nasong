


"""Auto-generated test stubs for core.values.basic.value_white_noise."""

import pytest
from unittest.mock import MagicMock, patch
import core.values.basic.value_white_noise


class TestWhiteNoise:
    """Tests for WhiteNoise."""

    def setup_method(self):
        """Create a fresh instance for each test."""
        # -- Setup Constructor Arguments --
        seed = 0
        scale = 0.0
        self.instance = core.values.basic.value_white_noise.WhiteNoise(seed, scale)

    def test_get_item(self):
        """Test for WhiteNoise.get_item."""
        # -- Setup --
        index = 0
        sample_rate = 0
        # -- Act --
        result = self.instance.get_item(index, sample_rate)
        # -- Assert --
        assert result == 0.0

    def test_vectorized_noise(self):
        """Test for WhiteNoise.vectorized_noise."""
        # -- Setup --
        indexes_buffer = 0.0
        seed = 0
        scale = 0.0
        # mock_astype = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.vectorized_noise(indexes_buffer, seed, scale)
        # -- Assert --
        assert result == 0.0

    def test_getitem_np(self):
        """Test for WhiteNoise.getitem_np."""
        # -- Setup --
        indexes_buffer = 0.0
        sample_rate = 0
        # mock_vectorized_noise = MagicMock(return_value=0.0)
        # -- Act --
        result = self.instance.getitem_np(indexes_buffer, sample_rate)
        # -- Assert --
        assert result == 0.0

    def test_getitem_torch(self):
        """Test for WhiteNoise.getitem_torch."""
        # -- Setup --
        indexes_buffer = None
        sample_rate = 0
        device = ""
        # mock_to = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.getitem_torch(indexes_buffer, sample_rate, device)
        # -- Assert --
        assert result == None

    def test_backward(self):
        """Test for WhiteNoise.backward."""
        # -- Setup --
        grad_output = 0.0
        context = ""
        sample_rate = 0
        # -- Act --
        result = self.instance.backward(grad_output, context, sample_rate)
        # -- Assert --
        assert result == None
