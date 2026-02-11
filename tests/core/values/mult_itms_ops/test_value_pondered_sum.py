


"""Auto-generated test stubs for core.values.mult_itms_ops.value_pondered_sum."""

import pytest
from unittest.mock import MagicMock, patch
import core.values.mult_itms_ops.value_pondered_sum


class TestPonderedSum:
    """Tests for PonderedSum."""

    def setup_method(self):
        """Create a fresh instance for each test."""
        # -- Setup Constructor Arguments --
        values = []
        self.instance = core.values.mult_itms_ops.value_pondered_sum.PonderedSum(values)

    def test_get_item(self):
        """Test for PonderedSum.get_item."""
        # -- Setup --
        index = 0
        sample_rate = 0
        # mock_get_item = MagicMock(return_value=0.0)
        # -- Act --
        result = self.instance.get_item(index, sample_rate)
        # -- Assert --
        assert result == 0.0

    def test_getitem_np(self):
        """Test for PonderedSum.getitem_np."""
        # -- Setup --
        indexes_buffer = 0.0
        sample_rate = 0
        # mock_zeros_like = MagicMock(return_value=None)
        # mock_getitem_np = MagicMock(return_value=0.0)
        # -- Act --
        result = self.instance.getitem_np(indexes_buffer, sample_rate)
        # -- Assert --
        assert result == 0.0

    def test_getitem_torch(self):
        """Test for PonderedSum.getitem_torch."""
        # -- Setup --
        indexes_buffer = None
        sample_rate = 0
        device = ""
        # mock_zeros_like = MagicMock(return_value=None)
        # mock_getitem_torch = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.getitem_torch(indexes_buffer, sample_rate, device)
        # -- Assert --
        assert result == None

    def test_backward(self):
        """Test for PonderedSum.backward."""
        # -- Setup --
        grad_output = 0.0
        context = ""
        sample_rate = 0
        # mock_getitem_np = MagicMock(return_value=0.0)
        # mock_backward = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.backward(grad_output, context, sample_rate)
        # -- Assert --
        assert result == None
