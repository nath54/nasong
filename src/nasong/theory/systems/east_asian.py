# Copyright (C) 2026 Nathan Cerisara <https://github.com/nath54/nasong>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""East Asian music theory approximations.

This module provides approximations of traditional East Asian scales, including
the Japanese 'Yo' (bright) and 'In' (dark/Sakura) pentatonic systems.
"""

#
### Import Modules. ###
#
from nasong.theory.core.scale import Scale
from nasong.theory.core.pitch import Note
from nasong.theory.core.interval import Interval


class EastAsian:
    """Namespace for East Asian musical scale factories (Japanese, Chinese)."""

    # Pentatonic scales are fundamental.
    # Chinese Major Pentatonic: 1, 2, 3, 5, 6 (Same as African/Western Major Pentatonic)
    # Japanese:
    # - Yo scale (Bright): 1, 2, 4, 5, 7? No. Yo is 2, 3, 2, 2, 3 intervals?
    # Yo scale (Ascending): C D F G A C (1, 2, 4, 5, 6) -> Intervals: 2, 3, 2, 2, 3
    # - In scale (Sakura/Dark): 1, b2, 4, 5, b6
    # In scale (Ascending): C Db F G Ab C -> Intervals: 1, 4, 2, 1, 4

    @staticmethod
    def yo_scale(root: str) -> Scale:
        """Generates a Japanese 'Yo' (bright) pentatonic scale."""
        intervals = [Interval(i) for i in [2, 3, 2, 2, 3]]
        return Scale(Note(root), intervals, name="Yo Scale")

    @staticmethod
    def in_scale(root: str) -> Scale:
        """Generates a Japanese 'In' (dark/Sakura) pentatonic scale."""
        intervals = [Interval(i) for i in [1, 4, 2, 1, 4]]
        return Scale(Note(root), intervals, name="In Scale")
