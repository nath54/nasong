from nasong.theory.core.scale import Scale
from nasong.theory.structures.progression import Progression
from nasong.theory.structures.chord import Chord
from nasong.theory.core.time import Duration, QUARTER
from nasong.theory.systems.western import Western
from typing import List
import random


class Jazz:
    """
    Generates Jazz-style progressions and voicings.
    """

    @staticmethod
    def ii_V_I(root: str = "C4", minor: bool = False) -> Progression:
        """
        Creates a classic ii-V-I progression.
        """
        scale_type = "minor" if minor else "major"
        scale = Scale.from_name(root, scale_type)

        # ii-V-I
        # Major: ii7 - V7 - Imaj7
        # Minor: ii7b5 - V7alt - imin7

        if minor:
            # Basic minor ii-V-i
            qualities = ["min7b5", "dom7", "min7"]
            degrees = [
                "ii",
                "V",
                "i",
            ]  # Roman numerals might need parsing support for lower case
            # Current implementation supports basic strings
            prog = Progression.from_roman_numerals(scale, degrees, duration=QUARTER)
            # Adjust qualities manually if needed, or rely on scale logic
            # This is a placeholder for more advanced logic
            return prog
        else:
            return Progression.from_roman_numerals(
                scale, ["ii", "V", "I"], duration=QUARTER
            )

    @staticmethod
    def generate_random_standards_progression(length: int = 4) -> Progression:
        """
        Generates a random sequence using common jazz turnarounds.
        """
        # Placeholder
        return Jazz.ii_V_I()
