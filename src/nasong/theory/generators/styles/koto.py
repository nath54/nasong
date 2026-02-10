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
from nasong.theory.core.time import QUARTER
from nasong.theory.systems.east_asian import EastAsian
from nasong.theory.structures.progression import Progression


class Koto:
    """
    Generates Koto-style melodic patterns.
    """

    @staticmethod
    def traditional_motif(root: str = "D4"):
        """
        Generates a motif using the In scale.
        """
        scale = EastAsian.in_scale(root)

        # Example pattern: Root -> 5th -> 4th -> Root
        # Indices: 0, 3, 2, 0
        from nasong.theory.structures.chord import Chord

        if hasattr(scale, "notes"):
            # Indices modulo length
            notes = [scale.notes[i % len(scale.notes)] for i in [0, 3, 2, 0]]
        else:
            # Fallback
            notes = []
        # Create single-note chords
        chords = [Chord(root=n, intervals=[], name="Note") for n in notes]

        return Progression(chords, [QUARTER] * 4)
