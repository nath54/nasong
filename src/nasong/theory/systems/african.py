from typing import List, Tuple
from nasong.theory.core.scale import Scale
from nasong.theory.core.pitch import Note
from nasong.theory.core.interval import Interval
from nasong.theory.structures.rhythm import Rhythm


class African:
    """
    Utilities for African musical concepts.
    """

    # 5-tone scales are common (Pentatonic)
    # E.g., Major Pentatonic: 1, 2, 3, 5, 6
    # Relative intervals: 2, 2, 3, 2, 3

    @staticmethod
    def pentatonic(root: str) -> Scale:
        """
        Standard major pentatonic often found in West African music.
        """
        intervals = [Interval(i) for i in [2, 2, 3, 2, 3]]
        return Scale(Note(root), intervals, name="African Pentatonic")

    @staticmethod
    def polyrhythm(
        ratio: Tuple[int, int], length: int = 12
    ) -> Tuple[List[int], List[int]]:
        """
        Generates two rhythmic patterns representing a polyrhythm (e.g., 3:2).
        Returns two lists of onset indices.
        """
        # Simple Euclidean-like distribution or just pulse markers
        pulse_a = ratio[0]
        pulse_b = ratio[1]

        # Taking LCM length or provided length
        # For 3:2 in 12 steps:
        # A (3 beats): 0, 4, 8 (Every 4 steps) -> 3 impacts
        # B (2 beats): 0, 6 (Every 6 steps) -> 2 impacts

        step_a = length / pulse_a
        step_b = length / pulse_b

        rhythm_a = [int(i * step_a) for i in range(pulse_a)]
        rhythm_b = [int(i * step_b) for i in range(pulse_b)]

        return rhythm_a, rhythm_b
