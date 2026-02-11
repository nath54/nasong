


"""Auto-generated test stubs for theory.core.interval."""

import pytest
from unittest.mock import MagicMock, patch
import theory.core.interval


class TestInterval:
    """Tests for Interval."""

    def setup_method(self):
        """Create a fresh instance for each test."""
        # -- Setup Constructor Arguments --
        value = 0
        self.instance = theory.core.interval.Interval(value)

    def test__parse_name(self):
        """Test for Interval._parse_name."""
        # -- Setup --
        name = ""
        # mock_ValueError = MagicMock(return_value=None)
        # -- Act --
        result = self.instance._parse_name(name)
        # -- Assert --
        assert result == 0

    def test_add_to(self):
        """Test for Interval.add_to."""
        # -- Setup --
        pitch = None
        # mock_TypeError = MagicMock(return_value=None)
        # mock_is_integer = MagicMock(return_value=None)
        # mock_transpose = MagicMock(return_value=None)
        # mock_Hz = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.add_to(pitch)
        # -- Assert --
        assert result == None
