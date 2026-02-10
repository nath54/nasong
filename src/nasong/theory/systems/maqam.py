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


"""
TODO: add full docstring, explaining what the goal of this script is, and explaining for each class and each function what is it, how it works, and how to use it.
"""

#
### Import Modules. ###
#
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
