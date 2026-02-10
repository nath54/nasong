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
from nasong.theory.structures.progression import Progression


class Salsa:
    """
    Generates Salsa patterns and montunos.
    """

    @staticmethod
    def montuno_progression(root: str = "G4", minor: bool = True) -> Progression:
        """
        Typical simple montuno: i - V
        """
        scale = Western.minor(root) if minor else Western.major(root)
        return Progression.from_roman_numerals(scale, ["i", "V"])

    @staticmethod
    def clave_rhythm(direction: str = "2-3"):
        """
        Returns a Clave rhythm pattern.
        """
        # 2-3 Son Clave: ..X.X... .X..X...
        # 3-2 Son Clave: X..X..X. ..X.X...
        pass
