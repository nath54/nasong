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
from nasong.theory.core.pitch import Note
from nasong.theory.core.scale import Scale
from nasong.theory.core.interval import Interval


class Raga:
    """
    Approximation of Indian Classical Ragas.
    """

    # Definitions of Thaats (Parent Scales) as offsets from root
    # Bilawal (Major): 0 2 4 5 7 9 11
    # Kalyan (Lydian): 0 2 4 6 7 9 11
    # Khamaj (Mixolydian): 0 2 4 5 7 9 10
    # Bhairav: 0 1 4 5 7 8 11
    # Purvi: 0 1 4 6 7 8 11
    # Marwa: 0 1 4 6 7 9 11
    # Kafi (Dorian): 0 2 3 5 7 9 10
    # Asavari (Aeolian): 0 2 3 5 7 8 10
    # Bhairavi (Phrygian): 0 1 3 5 7 8 10
    # Todi: 0 1 3 6 7 8 11

    PATTERNS = {
        "bilawal": [2, 2, 1, 2, 2, 2, 1],
        "kalyan": [2, 2, 2, 1, 2, 2, 1],
        "khamaj": [2, 2, 1, 2, 2, 1, 2],
        "bhairav": [1, 3, 1, 2, 1, 3, 1],
        "purvi": [1, 3, 2, 1, 1, 3, 1],
        "marwa": [1, 3, 2, 1, 2, 2, 1],
        "kafi": [2, 1, 2, 2, 2, 1, 2],
        "asavari": [2, 1, 2, 2, 1, 2, 2],
        "bhairavi": [1, 2, 2, 2, 1, 2, 2],
        "todi": [1, 2, 3, 1, 1, 3, 1],
    }

    @staticmethod
    def create(root: str, thaat_name: str) -> Scale:
        if thaat_name.lower() in Raga.PATTERNS:
            intervals = [Interval(s) for s in Raga.PATTERNS[thaat_name.lower()]]
            return Scale(Note(root), intervals, name=f"Raga {thaat_name}")
        raise ValueError(f"Unknown Thaat: {thaat_name}")
