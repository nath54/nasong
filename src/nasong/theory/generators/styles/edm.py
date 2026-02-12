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


"""EDM (Electronic Dance Music) style generators.

This module provides generators for high-energy EDM components, including
'Epic' chord vamps and standard dance floor beats.
"""

#
### Import Modules. ###
#
from nasong.theory.systems.western import Western
from nasong.theory.structures.rhythm import Rhythm, four_on_the_floor
from nasong.theory.structures.progression import Progression


class EDM:
    """Generates common EDM chord progressions and rhythmic foundations."""

    @staticmethod
    def epic_chords(root: str = "F4") -> Progression:
        """Generates an 'Epic' dance progression (e.g., vi-IV-I-V).

        Args:
            root (str): The root key. Defaults to "F4".

        Returns:
            Progression: A sequence of powerful chords.
        """
        scale = Western.major(root)
        return Progression.from_roman_numerals(scale, ["vi", "IV", "I", "V"])

    @staticmethod
    def basic_beat() -> Rhythm:
        """Returns a generic 'four on the floor' kick rhythm."""
        return four_on_the_floor()
