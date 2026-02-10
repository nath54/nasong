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
from nasong.theory.systems.western import Western
from nasong.theory.structures.rhythm import four_on_the_floor
from nasong.theory.structures.progression import Progression


class EDM:
    """
    Generates EDM components.
    """

    @staticmethod
    def epic_chords(root: str = "F4") -> Progression:
        """
        Generates an 'Epic' progression (e.g. vi - IV - I - V).
        """
        scale = Western.major(root)
        return Progression.from_roman_numerals(scale, ["vi", "IV", "I", "V"])

    @staticmethod
    def basic_beat():
        return four_on_the_floor()
