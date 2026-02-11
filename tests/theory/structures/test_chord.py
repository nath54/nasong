


"""Auto-generated test stubs for theory.structures.chord."""

import pytest
from unittest.mock import MagicMock, patch
import theory.structures.chord


class TestChord:
    """Tests for Chord."""

    def setup_method(self):
        """Create a fresh instance for each test."""
        self.instance = theory.structures.chord.Chord()

    def test_pitches(self):
        """Test for Chord.pitches."""
        # -- Setup --
        # mock_add_to = MagicMock(return_value=None)
        # mock_append = MagicMock(return_value=None)
        # mock_pop = MagicMock(return_value=None)
        # mock_transpose = MagicMock(return_value=None)
        # mock_Hz = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.pitches()
        # -- Assert --
        assert result == []

    def test_notes(self):
        """Test for Chord.notes."""
        # -- Setup --
        # mock_Note = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.notes()
        # -- Assert --
        assert result == []

    def test_from_name(self):
        """Test for Chord.from_name."""
        # -- Setup --
        root = ""
        quality = ""
        duration = None
        # mock_CoreNote = MagicMock(return_value=None)
        # mock_get = MagicMock(return_value=None)
        # mock_cls = MagicMock(return_value=None)
        # mock_lower = MagicMock(return_value=None)
        # mock_ValueError = MagicMock(return_value=None)
        # mock_Interval = MagicMock(return_value=None)
        # -- Act --
        result = self.instance.from_name(root, quality, duration)
        # -- Assert --
        assert result == None
