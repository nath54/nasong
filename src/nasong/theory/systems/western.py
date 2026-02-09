"""
Western Music System Defaults.
"""

from nasong.theory.core.scale import Scale, _SCALE_PATTERNS
from nasong.theory.core.pitch import Note, Tuning


class Western:
    """
    Namespace for Western music theory constants and factories.
    """

    # Common Scales factories
    @staticmethod
    def major(root: str) -> Scale:
        return Scale.from_name(root, "major")

    @staticmethod
    def minor(root: str) -> Scale:
        return Scale.from_name(root, "minor")

    @staticmethod
    def dorian(root: str) -> Scale:
        return Scale.from_name(root, "dorian")

    @staticmethod
    def phrygian(root: str) -> Scale:
        return Scale.from_name(root, "phrygian")

    @staticmethod
    def lydian(root: str) -> Scale:
        return Scale.from_name(root, "lydian")

    @staticmethod
    def mixolydian(root: str) -> Scale:
        return Scale.from_name(root, "mixolydian")

    @staticmethod
    def locrian(root: str) -> Scale:
        return Scale.from_name(root, "locrian")

    @staticmethod
    def aeolian(root: str) -> Scale:
        return Scale.from_name(root, "aeolian")
