from nasong.theory.core.scale import Scale
from nasong.theory.structures.progression import Progression
from nasong.theory.structures.rhythm import four_on_the_floor
from nasong.theory.systems.western import Western


class EDM:
    """
    Generates EDM components.
    """

    @staticmethod
    def epic_chords(root: str = "F4") -> Progression:
        """
        Generates an 'Epic' progression (e.g. vi - IV - I - V).
        """
        scale = Western.major(root)
        return Progression.from_roman_numerals(scale, ["vi", "IV", "I", "V"])

    @staticmethod
    def basic_beat():
        return four_on_the_floor()
