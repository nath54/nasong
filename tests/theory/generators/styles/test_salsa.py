"""Tests for theory.generators.styles.salsa."""

from nasong.theory.generators.styles.salsa import Salsa
from nasong.theory.structures.rhythm import Rhythm
from nasong.theory.structures.progression import Progression


class TestSalsa:
    """Tests for Salsa."""

    def test_montuno_progression_minor(self):
        """Test montuno_progression returns a valid minor Progression."""
        result = Salsa.montuno_progression("G4", minor=True)
        assert isinstance(result, Progression)
        assert len(result.chords) == 2

    def test_montuno_progression_major(self):
        """Test montuno_progression returns a valid major Progression."""
        result = Salsa.montuno_progression("C4", minor=False)
        assert isinstance(result, Progression)
        assert len(result.chords) == 2

    def test_clave_rhythm_2_3(self):
        """Test clave_rhythm returns a Rhythm for 2-3 direction."""
        result = Salsa.clave_rhythm("2-3")
        assert isinstance(result, Rhythm)
        assert len(result.events) > 0

    def test_clave_rhythm_3_2(self):
        """Test clave_rhythm returns a Rhythm for 3-2 direction."""
        result = Salsa.clave_rhythm("3-2")
        assert isinstance(result, Rhythm)
        assert len(result.events) > 0

    def test_clave_rhythm_default_is_2_3(self):
        """Test that the default clave direction is 2-3."""
        result = Salsa.clave_rhythm()
        assert isinstance(result, Rhythm)
