from nasong.theory.core.scale import Scale
from nasong.theory.structures.progression import Progression
from nasong.theory.systems.western import Western
from nasong.theory.structures.rhythm import Rhythm


class Celtic:
    """
    Generates Celtic/Folk patterns.
    """

    @staticmethod
    def jig_rhythm() -> Rhythm:
        """
        Compound time (6/8).
        ONE-two-three TWO-two-three
        """
        # Placeholder for specific rhythm generator
        pass

    @staticmethod
    def dorian_tune(root: str = "D4") -> Progression:
        """
        Dorian mode is very common in Celtic music.
        i - VII - i - IV
        """
        scale = Western.mode(root, "dorian")
        return Progression.from_roman_numerals(scale, ["i", "VII", "i", "IV"])
