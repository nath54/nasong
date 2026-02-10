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
from nasong.theory.systems.african import African
from nasong.theory.structures.progression import Progression


class Afrobeat:
    """
    Generates Afrobeat style patterns.
    """

    @staticmethod
    def polyrhythmic_groove(root: str = "C4"):
        """
        Generates a basic 3:2 polyrhythm structure.
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
