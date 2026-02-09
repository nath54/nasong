from nasong.theory.core.scale import Scale
from nasong.theory.structures.progression import Progression
from nasong.theory.systems.african import African
from nasong.theory.structures.rhythm import Rhythm
from nasong.theory.core.time import QUARTER


class Afrobeat:
    """
    Generates Afrobeat style patterns.
    """

    @staticmethod
    def polyrhythmic_groove(root: str = "C4"):
        """
        Generates a basic 3:2 polyrhythm structure.
        """
        scale = African.pentatonic(root)
        # 3 against 2
        r_a, r_b = African.polyrhythm((3, 2), length=12)

        # TODO: Return actual Rhythm objects with notes mapped from scale
        # For now, returning the raw onset data structure or a placeholder progression

        # A progression that stays on the root but has complex rhythm?
        # Or just a simple I-IV vamp
        prog = Progression.from_roman_numerals(scale, ["I", "IV"], duration=QUARTER * 4)
        return prog
