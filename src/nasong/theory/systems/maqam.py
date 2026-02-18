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


"""Arabic Maqam system approximations.

This module provides approximations of the Arabic Maqam system using 24-Tone
Equal Temperament (quarter tones). It includes standard patterns for common
Maqamat like Rast and Bayati.
"""

#
### Import Modules. ###
#
from nasong.theory.core.scale import Scale
from nasong.theory.core.pitch import Note
from nasong.theory.core.interval import Interval


class Maqam:
    """Approximation of Arabic Maqam system using microtones (quarter tones)."""

    # Assuming 24-TET for approximations.
    # Half sharp = 0.5 (50 cents)
    # 1 step = 2 semitones = 200 cents
    # 3/4 step = 1.5 semitones = 150 cents

    PATTERNS: dict[str, list[int | float]] = {
        "ajam": [2, 2, 1, 2, 2, 2, 1],  # Major
        "nahawand": [2, 1, 2, 2, 1, 2, 2],  # Minor
        "rast": [2, 1.5, 1.5, 2, 2, 1.5, 1.5],  # Approx
        "bayati": [1.5, 1.5, 2, 2, 1, 2, 2],  # Approx
        "hijaz": [1, 3, 1, 2, 1, 2, 2],  # Phrygian Dominant-ish
    }

    @staticmethod
    def create(root: str, maqam_name: str) -> Scale:
        """Creates a Maqam scale object based on a pattern name.

        Args:
            root (str): The root note name (e.g., "C4").
            maqam_name (str): The name of the Maqam (e.g., "rast", "bayati").

        Returns:
            Scale: The resulting Scale object with microtonal intervals.

        Raises:
            ValueError: If the Maqam name is unknown.
        """
        if maqam_name.lower() in Maqam.PATTERNS:
            intervals = [Interval(s) for s in Maqam.PATTERNS[maqam_name.lower()]]
            return Scale(Note(root), intervals, name=f"Maqam {maqam_name}")
        raise ValueError(f"Unknown Maqam: {maqam_name}")
