"""
Western Music System Defaults.
"""

from nasong.theory.core.scale import Scale, _SCALE_PATTERNS
from nasong.theory.core.pitch import Note, Tuning


class WesternMeta(type):
    """
    Metaclass to support dynamic note access (e.g., Western.C4).
    """

    def __getattr__(cls, name):
        try:
            return Note(name)
        except ValueError:
            raise AttributeError(
                f"type object '{cls.__name__}' has no attribute '{name}'"
            )


class Western(metaclass=WesternMeta):
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
    def mode(root: str, mode_name: str) -> Scale:
        return Scale.from_name(root, mode_name)
