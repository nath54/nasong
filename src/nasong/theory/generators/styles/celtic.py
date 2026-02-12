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


"""Celtic and Folk style generators.

This module provides generators for Celtic-style jigs and modal tunes,
primarily focusing on compound meters and Dorian-based melodies.
"""

#
### Import Modules. ###
#
from nasong.theory.systems.western import Western
from nasong.theory.structures.rhythm import Rhythm
from nasong.theory.structures.progression import Progression


class Celtic:
    """Generates Celtic/Folk melodic patterns and rhythmic jigs."""

    @staticmethod
    def jig_rhythm() -> Rhythm:
        """Generates a standard compound time (6/8) jig rhythm."""
        # Placeholder for specific rhythm generator
        pass

    @staticmethod
    def dorian_tune(root: str = "D4") -> Progression:
        """Generates a Dorian-mode folk progression (e.g., i-VII-i-IV).

        Args:
            root (str): The fundamental note of the tune. Defaults to "D4".

        Returns:
            Progression: The resulting melodic progression.
        """
        scale = Western.mode(root, "dorian")
        return Progression.from_roman_numerals(scale, ["i", "VII", "i", "IV"])
