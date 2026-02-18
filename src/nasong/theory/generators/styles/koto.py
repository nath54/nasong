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


"""Japanese Koto-style generators.

This module provides generators for traditional Japanese melodic patterns,
specifically targeting the 'In' pentatonic scale and Koto-like motifs.
"""

#
### Import Modules. ###
#
from nasong.theory.core.time import QUARTER
from nasong.theory.systems.east_asian import EastAsian
from nasong.theory.structures.progression import Progression
from nasong.theory.structures.chord import Chord


class Koto:
    """Generates traditional Japanese Koto-style melodic patterns."""

    @staticmethod
    def traditional_motif(root: str = "D4") -> Progression:
        """Generates a simple ascending/descending motif using the 'In' scale."""
        scale = EastAsian.in_scale(root)

        # Example pattern: Root -> 5th -> 4th -> Root
        # Indices: 0, 3, 2, 0

        if hasattr(scale, "notes"):
            # Indices modulo length
            notes = [scale.notes[i % len(scale.notes)] for i in [0, 3, 2, 0]]
        else:
            # Fallback
            notes = []
        # Create single-note chords
        chords = [Chord(root=n, intervals=[], name="Note") for n in notes]

        return Progression(chords, [QUARTER] * 4)
