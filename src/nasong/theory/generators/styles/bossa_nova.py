from nasong.theory.core.scale import Scale
from nasong.theory.structures.progression import Progression
from nasong.theory.systems.western import Western
from nasong.theory.core.time import QUARTER


class BossaNova:
    """
    Generates Bossa Nova progressions and rhythms.
    """

    @staticmethod
    def standard_progression(root: str = "C4") -> Progression:
        """
        Classic Bossa progression: Imaj7 - II7 - ii7 - V7
        (Or similar turnarounds)
        """
        scale = Western.major(root)
        # Using Roman Numeral placeholder
        # Ideally needs extensions: Imaj7, II7 (Major with minor 7th? No, Dom7), ii7, V7b9 etc.
        # This requires extendable parser or manual Chord construction.
        # For prototype, using basic numerals which map to diatonic chords usually.
        # We can force chromatic chords by name?
        # Let's use scale degrees.
        return Progression.from_roman_numerals(scale, ["I", "II", "ii", "V"])
