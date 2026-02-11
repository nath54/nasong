


"""Auto-generated test stubs for core.values.single_itms_ops.value_basic_scaling."""

import pytest
from unittest.mock import MagicMock, patch
import core.values.single_itms_ops.value_basic_scaling


class TestBasicScaling:
    """Tests for BasicScaling."""

    def setup_method(self):
        """Create a fresh instance for each test."""
        # -- Setup Constructor Arguments --
        value = None
        mult_scale = None
        sum_scale = None
        self.instance = core.values.single_itms_ops.value_basic_scaling.BasicScaling(value, mult_scale, sum_scale)

    def test_get_item(self):
        """Test for BasicScaling.get_item."""
        # -- Setup --
        index = 0
        sample_rate = 0
        # mock_get_item = MagicMock(return_value=0.0)
        # -- Act --
        result = self.instance.get_item(index, sample_rate)
        # -- Assert --
        assert result == 0.0

    def test_getitem_np(self):
        """Test for BasicScaling.getitem_np."""
        # -- Setup --
        indexes_buffer = 0.0
        sample_rate = 0
        # mock_getitem_np = MagicMock(return_value=0.0)
        # mock_multiply = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.getitem_np(indexes_buffer, sample_rate)
        # -- Assert --
        assert result == 0.0

    def test_getitem_torch(self):
        """Test for BasicScaling.getitem_torch."""
        # -- Setup --
        indexes_buffer = None
        sample_rate = 0
        device = ""
        # mock_getitem_torch = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.getitem_torch(indexes_buffer, sample_rate, device)
        # -- Assert --
        assert result == None

    def test_backward(self):
        """Test for BasicScaling.backward."""
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
