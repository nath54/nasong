"""Auto-generated test stubs for theory.generators.styles.koto."""

import theory.generators.styles.koto


class TestKoto:
    """Tests for Koto."""

    def setup_method(self):
        """Create a fresh instance for each test."""
        self.instance = theory.generators.styles.koto.Koto()

    def test_traditional_motif(self):
        """Test for Koto.traditional_motif."""
        # -- Setup --
        root = "D4"
        # -- Act --
        result = self.instance.traditional_motif(root)
        # -- Assert --
        assert result is not None
        assert len(result.chords) == 4
        assert result.scale.name == "In Scale"
