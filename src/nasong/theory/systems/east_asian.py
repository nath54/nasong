from typing import List, Tuple
from nasong.theory.core.scale import Scale
from nasong.theory.core.pitch import Note
from nasong.theory.core.interval import Interval


class EastAsian:
    """
    Approximations of East Asian scales (Japanese, Chinese).
    """

    # Pentatonic scales are fundamental.
    # Chinese Major Pentatonic: 1, 2, 3, 5, 6 (Same as African/Western Major Pentatonic)
    # Japanese:
    # - Yo scale (Bright): 1, 2, 4, 5, 7? No. Yo is 2, 3, 2, 2, 3 intervals?
    # Yo scale (Ascending): C D F G A C (1, 2, 4, 5, 6) -> Intervals: 2, 3, 2, 2, 3
    # - In scale (Sakura/Dark): 1, b2, 4, 5, b6
    # In scale (Ascending): C Db F G Ab C -> Intervals: 1, 4, 2, 1, 4

    @staticmethod
    def yo_scale(root: str) -> Scale:
        """
        'Bright' pentatonic scale (e.g. Folk music).
        Intervals: 2, 3, 2, 2, 3
        """
        intervals = [Interval(i) for i in [2, 3, 2, 2, 3]]
        return Scale(Note(root), intervals, name="Yo Scale")

    @staticmethod
    def in_scale(root: str) -> Scale:
        """
        'Dark' or 'Sakura' pentatonic scale.
        Intervals: 1, 4, 2, 1, 4
        """
        intervals = [Interval(i) for i in [1, 4, 2, 1, 4]]
        return Scale(Note(root), intervals, name="In Scale")
