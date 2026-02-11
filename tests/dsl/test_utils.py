


"""Auto-generated test stubs for dsl.utils."""

import pytest
from unittest.mock import MagicMock, patch
import dsl.utils


def test_arpeggiate():
    """Test for arpeggiate."""
    # -- Setup --
    chord = None
    pattern = []
    duration = None
    # mock_Duration = MagicMock(return_value=None)
    # mock_append = MagicMock(return_value=None)
    # mock_Note = MagicMock(return_value=None)
    # mock_transpose = MagicMock(return_value=None)
    # -- Act --
    result = dsl.utils.arpeggiate(chord, pattern, duration)
    # -- Assert --
    assert result == []
