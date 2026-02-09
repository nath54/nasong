from typing import List
from nasong.theory.core.scale import Scale
from nasong.theory.core.pitch import Note
from nasong.theory.core.interval import Interval


class Maqam:
    """
    Approximation of Arabic Maqam system using microtones (quarter tones).
    """

    # Assuming 24-TET for approximations.
    # Half sharp = 0.5 (50 cents)
    # 1 step = 2 semitones = 200 cents
    # 3/4 step = 1.5 semitones = 150 cents

    # Bayati: 0, 1.5, 3, 5, 7, 8, 10
    # Intervals in semitones: 1.5, 1.5, 2, 2, 1, 2, 2 (Approximated)
    # Actually:
    # Root -> 2nd (Half flat): ~1.5 semitones?
    # No, Half flat 2nd is ~1.5 semitones from root?
    # Let's use standard definitions:
    # Rast: C, D0.5b, E0.5b, F, G, A0.5b, B0.5b
    # Intervals: 2, 1.75, 1.25, 2, 1.75, 1.25 ?
    # Let's use float semitones.

    # Common Ajnas (sets):
    # Ajam (Major-like): 2, 2, 1
    # Nahawand (Minor-like): 2, 1, 2
    # Rast: 2, 1.5, 1.5 (approx neutral third)
    # Bayati: 1.5, 1.5, 2
    # Hijaz: 1, 3, 1

    PATTERNS = {
        "ajam": [2, 2, 1, 2, 2, 2, 1],  # Major
        "nahawand": [2, 1, 2, 2, 1, 2, 2],  # Minor
        "rast": [2, 1.5, 1.5, 2, 2, 1.5, 1.5],  # Approx
        "bayati": [1.5, 1.5, 2, 2, 1, 2, 2],  # Approx
        "hijaz": [1, 3, 1, 2, 1, 2, 2],  # Phrygian Dominant-ish
    }

    @staticmethod
    def create(root: str, maqam_name: str) -> Scale:
        if maqam_name.lower() in Maqam.PATTERNS:
            intervals = [Interval(s) for s in Maqam.PATTERNS[maqam_name.lower()]]
            return Scale(Note(root), intervals, name=f"Maqam {maqam_name}")
        raise ValueError(f"Unknown Maqam: {maqam_name}")
