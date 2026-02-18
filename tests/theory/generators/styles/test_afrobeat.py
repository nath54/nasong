"""Tests for theory.generators.styles.afrobeat."""

from nasong.theory.generators.styles.afrobeat import Afrobeat
from nasong.theory.structures.progression import Progression
from nasong.theory.structures.rhythm import Rhythm


class TestAfrobeat:
    """Tests for Afrobeat."""

    def test_polyrhythmic_groove_returns_progression(self):
        """Test polyrhythmic_groove returns a Progression."""
        result = Afrobeat.polyrhythmic_groove("C4")
        assert isinstance(result, Progression)
        assert len(result.chords) == 2

    def test_polyrhythmic_groove_has_rhythm_a(self):
        """Test that rhythm_a is attached to the result."""
        result = Afrobeat.polyrhythmic_groove("C4")
        assert hasattr(result, "rhythm_a")
        assert isinstance(result.rhythm_a, Rhythm)  # type: ignore[attr-defined]

    def test_polyrhythmic_groove_has_rhythm_b(self):
        """Test that rhythm_b is attached to the result."""
        result = Afrobeat.polyrhythmic_groove("C4")
        assert hasattr(result, "rhythm_b")
        assert isinstance(result.rhythm_b, Rhythm)  # type: ignore[attr-defined]

    def test_polyrhythmic_groove_rhythm_events_count(self):
        """Test that rhythm events match the polyrhythm length (12 steps)."""
        result = Afrobeat.polyrhythmic_groove("C4")
        assert len(result.rhythm_a.events) == 12  # type: ignore[attr-defined]
        assert len(result.rhythm_b.events) == 12  # type: ignore[attr-defined]
