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


"""Indonesian Gamelan tuning approximations.

This module provides approximations of common Gamelan tuning systems, including
Slendro (equidistant 5-tone) and Pelog (non-equidistant 7-tone variations).
"""

#
### Import Modules. ###
#
from nasong.theory.core.pitch import Note
from nasong.theory.core.scale import Scale
from nasong.theory.core.interval import Interval


class Gamelan:
    """Approximation of Indonesian Gamelan tuning systems."""

    # Slendro: Roughly equidistant 5 tones per octave.
    # 12 / 5 = 2.4 semitones per step.

    # Pelog: Unequal 7 tones.

    PATTERNS = {
        "slendro": [2.4, 2.4, 2.4, 2.4, 2.4],
        "pelog_bem": [1, 2, 4, 1, 4],
        "pelog_barang": [2, 3, 2, 3, 2],
    }

    @staticmethod
    def create(root: str, type_name: str) -> Scale:
        """Creates a Gamelan scale based on a specific tuning type.

        Args:
            root (str): The root note name (e.g., "C4").
            type_name (str): The tuning system ("slendro", "pelog_bem",
                "pelog_barang").

        Returns:
            Scale: The resulting Scale object with microtonal or custom intervals.

        Raises:
            ValueError: If the Gamelan type is unknown.
        """
        if type_name.lower() in Gamelan.PATTERNS:
            intervals = [Interval(s) for s in Gamelan.PATTERNS[type_name.lower()]]
            return Scale(Note(root), intervals, name=f"Gamelan {type_name}")
        raise ValueError(f"Unknown Gamelan type: {type_name}")
