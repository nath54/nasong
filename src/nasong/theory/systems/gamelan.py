from typing import List
from nasong.theory.core.scale import Scale
from nasong.theory.core.pitch import Note
from nasong.theory.core.interval import Interval


class Gamelan:
    """
    Approximation of Indonesian Gamelan tuning systems.
    Slendro (5-tone) and Pelog (7-tone).
    """

    # Slendro: Roughly equidistant 5 tones per octave.
    # 12 / 5 = 2.4 semitones per step.

    # Pelog: Unequal 7 tones.
    # Approx intervals: 1.33, 2.66, ... very rough approx in 12-TET
    # Or strict Hz definitions?
    # For now, using rough semitone offsets.

    PATTERNS = {
        "slendro": [2.4, 2.4, 2.4, 2.4, 2.4],
        "pelog_bem": [1, 2, 4, 1, 4],  # Subset logic often used
        "pelog_barang": [2, 3, 2, 3, 2],
        # Full Pelog approx: 1.2, 1.5, ... hard to map to 12-TET names.
    }

    @staticmethod
    def create(root: str, type_name: str) -> Scale:
        if type_name.lower() in Gamelan.PATTERNS:
            intervals = [Interval(s) for s in Gamelan.PATTERNS[type_name.lower()]]
            return Scale(Note(root), intervals, name=f"Gamelan {type_name}")
        raise ValueError(f"Unknown Gamelan type: {type_name}")
