


"""Auto-generated test stubs for core.values.single_itms_ops.value_modulo."""

import pytest
from unittest.mock import MagicMock, patch
import core.values.single_itms_ops.value_modulo


class TestModulo:
    """Tests for Modulo."""

    def setup_method(self):
        """Create a fresh instance for each test."""
        # -- Setup Constructor Arguments --
        value = None
        modulo_value = None
        self.instance = core.values.single_itms_ops.value_modulo.Modulo(value, modulo_value)

    def test_get_item(self):
        """Test for Modulo.get_item."""
        # -- Setup --
        index = 0
        sample_rate = 0
        # mock_get_item = MagicMock(return_value=0.0)
        # -- Act --
        result = self.instance.get_item(index, sample_rate)
        # -- Assert --
        assert result == 0.0

    def test_getitem_np(self):
        """Test for Modulo.getitem_np."""
        # -- Setup --
        indexes_buffer = None
        sample_rate = 0
        # mock_getitem_np = MagicMock(return_value=None)
        # mock_where = MagicMock(return_value=None)
        # mock_mod = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.getitem_np(indexes_buffer, sample_rate)
        # -- Assert --
        assert result == None

    def test_getitem_torch(self):
        """Test for Modulo.getitem_torch."""
        # -- Setup --
        indexes_buffer = None
        sample_rate = 0
        device = None
        # mock_getitem_torch = MagicMock(return_value=None)
        # mock_where = MagicMock(return_value=None)
        # mock_fmod = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.getitem_torch(indexes_buffer, sample_rate, device)
        # -- Assert --
        assert result == None

    def test_backward(self):
        """Test for Modulo.backward."""
        # -- Setup --
        grad_output = None
        context = {}
        sample_rate = 0
        # mock_backward = MagicMock(return_value=None)
        # mock_zeros_like = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.backward(grad_output, context, sample_rate)
        # -- Assert --
        assert result == None
