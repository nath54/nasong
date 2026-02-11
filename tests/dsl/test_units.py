


"""Auto-generated test stubs for dsl.units."""

import pytest
from unittest.mock import MagicMock, patch
import dsl.units


class TestBPM:
    """Tests for BPM."""

    def setup_method(self):
        """Create a fresh instance for each test."""
        self.instance = dsl.units.BPM()

    def test_to_ms(self):
        """Test for BPM.to_ms."""
        # -- Setup --
        note_duration = 0.0
        # -- Act --
        result = self.instance.to_ms(note_duration)
        # -- Assert --
        assert result == 0.0

class TestMs:
    """Tests for Ms."""

    def setup_method(self):
        """Create a fresh instance for each test."""
        self.instance = dsl.units.Ms()

    def test_to_seconds(self):
        """Test for Ms.to_seconds."""
        # -- Setup --
        # -- Act --
        result = self.instance.to_seconds()
        # -- Assert --
        assert result == 0.0

class TestBars:
    """Tests for Bars."""

    def setup_method(self):
        """Create a fresh instance for each test."""
        self.instance = dsl.units.Bars()

class TestHz:
    """Tests for Hz."""

    def setup_method(self):
        """Create a fresh instance for each test."""
        self.instance = dsl.units.Hz()
