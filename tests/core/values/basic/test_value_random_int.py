


"""Auto-generated test stubs for core.values.basic.value_random_int."""

import pytest
from unittest.mock import MagicMock, patch
import core.values.basic.value_random_int


class TestRandomInt:
    """Tests for RandomInt."""

    def setup_method(self):
        """Create a fresh instance for each test."""
        # -- Setup Constructor Arguments --
        min_range = None
        max_range = None
        self.instance = core.values.basic.value_random_int.RandomInt(min_range, max_range)

    def test_get_item(self):
        """Test for RandomInt.get_item."""
        # -- Setup --
        index = 0
        sample_rate = 0
        # mock_randint = MagicMock(return_value=None)
        # mock_get_item = MagicMock(return_value=0.0)
        # -- Act --
        result = self.instance.get_item(index, sample_rate)
        # -- Assert --
        assert result == 0.0

    def test_getitem_np(self):
        """Test for RandomInt.getitem_np."""
        # -- Setup --
        indexes_buffer = 0.0
        sample_rate = 0
        # mock_getitem_np = MagicMock(return_value=0.0)
        # mock_astype = MagicMock(return_value=None)
        # mock_maximum = MagicMock(return_value=None)
        # mock_uniform = MagicMock(return_value=None)
        # mock_floor = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.getitem_np(indexes_buffer, sample_rate)
        # -- Assert --
        assert result == 0.0

    def test_getitem_torch(self):
        """Test for RandomInt.getitem_torch."""
        # -- Setup --
        indexes_buffer = None
        sample_rate = 0
        device = ""
        # mock_getitem_torch = MagicMock(return_value=None)
        # mock_rand_like = MagicMock(return_value=None)
        # mock_to = MagicMock(return_value=None)
        # mock_floor = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.getitem_torch(indexes_buffer, sample_rate, device)
        # -- Assert --
        assert result == None

    def test_backward(self):
        """Test for RandomInt.backward."""
        # -- Setup --
        grad_output = 0.0
        context = ""
        sample_rate = 0
        # mock_backward = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.backward(grad_output, context, sample_rate)
        # -- Assert --
        assert result == None
