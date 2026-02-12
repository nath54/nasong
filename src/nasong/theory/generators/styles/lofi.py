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


"""Lofi Hip Hop style generators.

This module provides generators for relaxed, atmospheric Lofi progressions,
characterized by extended jazz chords and slow, laid-back harmonic movement.
"""

#
### Import Modules. ###
#
from nasong.theory.systems.western import Western
from nasong.theory.structures.progression import Progression


class Lofi:
    """Generates chill, atmospheric Lofi Hip Hop chord progressions."""

    @staticmethod
    def chill_progression(root: str = "Db4") -> Progression:
        """Generates a relaxed ii-V-I (or similar) lofi progression.

        Args:
            root (str): The root key. Defaults to "Db4".

        Returns:
            Progression: The resulting atmospheric sequence.
        """
        # Simplified for now
        scale = Western.major(root)
        # Using Roman Numeral parser which defaults to basic triads/7ths based on scale
        # Ideally we'd specify extensions.
        return Progression.from_roman_numerals(scale, ["ii", "V", "I"])
