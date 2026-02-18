"""Tests for theory.generators.styles.jazz."""

from nasong.theory.generators.styles.jazz import Jazz
from nasong.theory.structures.progression import Progression


class TestJazz:
    """Tests for Jazz."""

    def test_ii_V_I_major(self):
        """Test ii-V-I returns a valid major Progression."""
        result = Jazz.ii_V_I("C4", minor=False)
        assert isinstance(result, Progression)
        assert len(result.chords) == 3

    def test_ii_V_I_minor(self):
        """Test ii-V-i returns a valid minor Progression."""
        result = Jazz.ii_V_I("C4", minor=True)
        assert isinstance(result, Progression)
        assert len(result.chords) == 3

    def test_generate_random_standards_progression_default(self):
        """Test random standards returns a Progression of default length."""
        result = Jazz.generate_random_standards_progression()
        assert isinstance(result, Progression)
        assert len(result.chords) == 4

    def test_generate_random_standards_progression_custom_length(self):
        """Test random standards respects the length parameter."""
        result = Jazz.generate_random_standards_progression(length=8)
        assert isinstance(result, Progression)
        assert len(result.chords) == 8
