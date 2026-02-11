


"""Auto-generated test stubs for core.values.complex.value_square."""

import pytest
from unittest.mock import MagicMock, patch
import core.values.complex.value_square


class TestSquare:
    """Tests for Square."""

    def setup_method(self):
        """Create a fresh instance for each test."""
        self.instance = core.values.complex.value_square.Square()

    def test_get_item(self):
        """Test for Square.get_item."""
        # -- Setup --
        index = 0
        sample_rate = 0
        # mock_get_item = MagicMock(return_value=0.0)
        # mock_floor = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.get_item(index, sample_rate)
        # -- Assert --
        assert result == 0.0

    def test_getitem_np(self):
        """Test for Square.getitem_np."""
        # -- Setup --
        indexes_buffer = None
        sample_rate = 0
        # mock_getitem_np = MagicMock(return_value=None)
        # mock_where = MagicMock(return_value=None)
        # mock_multiply = MagicMock(return_value=None)
        # mock_floor = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.getitem_np(indexes_buffer, sample_rate)
        # -- Assert --
        assert result == None

    def test_getitem_torch(self):
        """Test for Square.getitem_torch."""
        # -- Setup --
        indexes_buffer = None
        sample_rate = 0
        device = None
        # mock_getitem_torch = MagicMock(return_value=None)
        # mock_where = MagicMock(return_value=None)
        # mock_floor = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.getitem_torch(indexes_buffer, sample_rate, device)
        # -- Assert --
        assert result == None

    def test_backward(self):
        """Test for Square.backward."""
        # -- Setup --
        grad_output = None
        context = {}
        sample_rate = 0
        # mock_getitem_np = MagicMock(return_value=None)
        # mock_where = MagicMock(return_value=None)
        # mock_backward = MagicMock(return_value=None)
        # mock_floor = MagicMock(return_value=None)
        # mock_zeros_like = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.backward(grad_output, context, sample_rate)
        # -- Assert --
        assert result == None
