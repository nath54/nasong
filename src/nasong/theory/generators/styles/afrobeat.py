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


"""Afrobeat style generators.

This module provides algorithmic generators for Afrobeat-style patterns,
leveraging West African pentatonic scales and polyrhythmic structures.
"""

#
### Import Modules. ###
#
from nasong.theory.core.time import QUARTER
from nasong.theory.systems.african import African
from nasong.theory.structures.progression import Progression


class Afrobeat:
    """Generates Afrobeat style patterns and grooves."""

    @staticmethod
    def polyrhythmic_groove(root: str = "C4") -> Progression:
        """Generates a basic Afrobeat groove with a 3:2 polyrhythm structure.

        Args:
            root (str): The root note for the scale. Defaults to "C4".

        Returns:
            Progression: A rhythmic progression representing the groove.
        """
        scale = African.pentatonic(root)
        # 3 against 2
        r_a, r_b = African.polyrhythm((3, 2), length=12)

        # TODO: Return actual Rhythm objects with notes mapped from scale
        # For now, returning the raw onset data structure or a placeholder progression

        # A progression that stays on the root but has complex rhythm?
        # Or just a simple I-IV vamp
        prog = Progression.from_roman_numerals(scale, ["I", "IV"], duration=QUARTER * 4)
        return prog
