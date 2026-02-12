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


"""African musical theory concepts.

This module provides tools for working with common African musical structures,
including pentatonic scales and rhythmic polyrhythms.
"""

#
### Import Modules. ###
#
from nasong.theory.core.pitch import Note
from nasong.theory.core.scale import Scale
from nasong.theory.core.interval import Interval


class African:
    """Namespace for African musical concept factories and utilities."""

    # 5-tone scales are common (Pentatonic)
    # E.g., Major Pentatonic: 1, 2, 3, 5, 6
    # Relative intervals: 2, 2, 3, 2, 3

    @staticmethod
    def pentatonic(root: str) -> Scale:
        """Generates a Major Pentatonic scale commonly found in West African music."""
        intervals = [Interval(i) for i in [2, 2, 3, 2, 3]]
        return Scale(Note(root), intervals, name="African Pentatonic")

    @staticmethod
    def polyrhythm(
        ratio: tuple[int, int], length: int = 12
    ) -> tuple[list[int], list[int]]:
        """Generates pulse markers for a cross-rhythm (e.g., 3:2).

        Args:
            ratio (tuple[int, int]): The beat ratio (e.g., (3, 2)).
            length (int, optional): The total number of sub-division steps.
                Defaults to 12.

        Returns:
            tuple[list[int], list[int]]: Two lists of onset step indices.
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
