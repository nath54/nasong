


"""Auto-generated test stubs for core.values.mult_itms_ops.value_max."""

import pytest
from unittest.mock import MagicMock, patch
import core.values.mult_itms_ops.value_max


class TestMax:
    """Tests for Max."""

    def setup_method(self):
        """Create a fresh instance for each test."""
        # -- Setup Constructor Arguments --
        self.instance = core.values.mult_itms_ops.value_max.Max()

    def test_get_item(self):
        """Test for Max.get_item."""
        # -- Setup --
        index = 0
        sample_rate = 0
        # mock_get_item = MagicMock(return_value=0.0)
        # -- Act --
        result = self.instance.get_item(index, sample_rate)
        # -- Assert --
        assert result == 0.0

    def test_getitem_np(self):
        """Test for Max.getitem_np."""
        # -- Setup --
        indexes_buffer = 0.0
        sample_rate = 0
        # mock_stack = MagicMock(return_value=None)
        # mock_getitem_np = MagicMock(return_value=0.0)
        # -- Act --
        result = self.instance.getitem_np(indexes_buffer, sample_rate)
        # -- Assert --
        assert result == 0.0

    def test_getitem_torch(self):
        """Test for Max.getitem_torch."""
        # -- Setup --
        indexes_buffer = None
        sample_rate = 0
        device = ""
        # mock_getitem_torch = MagicMock(return_value=None)
        # mock_zeros_like = MagicMock(return_value=None)
        # mock_stack = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.getitem_torch(indexes_buffer, sample_rate, device)
        # -- Assert --
        assert result == None

    def test_backward(self):
        """Test for Max.backward."""
        # -- Setup --
        grad_output = 0.0
        context = ""
        sample_rate = 0
        # mock_stack = MagicMock(return_value=None)
        # mock_argmax = MagicMock(return_value=None)
        # mock_getitem_np = MagicMock(return_value=0.0)
        # mock_astype = MagicMock(return_value=None)
        # mock_backward = MagicMock(return_value=None)
        # mock_zeros_like = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.backward(grad_output, context, sample_rate)
        # -- Assert --
        assert result == None
