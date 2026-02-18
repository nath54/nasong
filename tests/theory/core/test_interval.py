"""Tests for theory.core.interval."""

import pytest

from nasong.theory.core.interval import Interval


class TestInterval:
    """Tests for Interval."""

    def test_from_semitones(self):
        """Test creating an Interval from an integer semitone count."""
        iv = Interval(7)
        assert iv.semitones == 7.0
        assert iv.ratio == pytest.approx(2 ** (7 / 12.0))

    def test_from_name_P5(self):
        """Test parsing 'P5' returns 7 semitones."""
        iv = Interval("P5")
        assert iv.semitones == 7.0

    def test_from_name_min3(self):
        """Test parsing 'min3' returns 3 semitones."""
        iv = Interval("min3")
        assert iv.semitones == 3.0

    def test_from_name_octave(self):
        """Test parsing 'octave' returns 12 semitones."""
        iv = Interval("octave")
        assert iv.semitones == 12.0

    def test_from_name_unknown_raises(self):
        """Test that an unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown interval name"):
            Interval("bad_name")

    def test_add_intervals(self):
        """Test adding two intervals."""
        a = Interval(3)
        b = Interval(4)
        result = a + b
        assert result.semitones == 7.0

    def test_negate_interval(self):
        """Test negating an interval."""
        iv = Interval(7)
        neg = -iv
        assert neg.semitones == -7.0
