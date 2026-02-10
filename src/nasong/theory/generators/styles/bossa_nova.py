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


class BossaNova:
    """
    Generates Bossa Nova progressions and rhythms.
    """

    @staticmethod
    def standard_progression(root: str = "C4") -> Progression:
        """
        Classic Bossa progression: Imaj7 - II7 - ii7 - V7
        (Or similar turnarounds)
        """
        scale = Western.major(root)
        #
        return Progression.from_roman_numerals(scale, ["I", "II", "ii", "V"])
