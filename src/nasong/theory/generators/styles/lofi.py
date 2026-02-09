from nasong.theory.core.scale import Scale
from nasong.theory.structures.progression import Progression
from nasong.theory.systems.western import Western


class Lofi:
    """
    Generates Lofi Hip Hop progressions.
    """

    @staticmethod
    def chill_progression(root: str = "Db4") -> Progression:
        """
        Often uses extended chords (7ths, 9ths) and slow tempos.
        Progression: ii9 - V13 - Imaj9
        """
        # Simplified for now
        scale = Western.major(root)
        # Using Roman Numeral parser which defaults to basic triads/7ths based on scale
        # Ideally we'd specify extensions.
        return Progression.from_roman_numerals(scale, ["ii", "V", "I"])
