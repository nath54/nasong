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
from nasong.theory.core.scale import Scale
from nasong.theory.core.time import QUARTER
from nasong.theory.structures.progression import Progression


class Jazz:
    """
    Generates Jazz-style progressions and voicings.
    """

    @staticmethod
    def ii_V_I(root: str = "C4", minor: bool = False) -> Progression:
        """
        Creates a classic ii-V-I progression.
        """
        scale_type = "minor" if minor else "major"
        scale = Scale.from_name(root, scale_type)

        # ii-V-I
        # Major: ii7 - V7 - Imaj7
        # Minor: ii7b5 - V7alt - imin7

        if minor:
            # Basic minor ii-V-i
            _qualities = ["min7b5", "dom7", "min7"]
            degrees = [
                "ii",
                "V",
                "i",
            ]
            #
            prog = Progression.from_roman_numerals(scale, degrees, duration=QUARTER)
            #
            return prog
        else:
            return Progression.from_roman_numerals(
                scale, ["ii", "V", "I"], duration=QUARTER
            )

    @staticmethod
    def generate_random_standards_progression(length: int = 4) -> Progression:
        """
        Generates a random sequence using common jazz turnarounds.
        """
        # Placeholder
        return Jazz.ii_V_I()
