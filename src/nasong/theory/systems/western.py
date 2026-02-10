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
from nasong.theory.core.pitch import Note
from nasong.theory.core.scale import Scale


class WesternMeta(type):
    """
    Metaclass to support dynamic note access (e.g., Western.C4).
    """

    def __getattr__(cls, name):
        try:
            return Note(name)
        except ValueError:
            raise AttributeError(
                f"type object '{cls.__name__}' has no attribute '{name}'"
            )


class Western(metaclass=WesternMeta):
    """
    Namespace for Western music theory constants and factories.
    """

    # Common Scales factories
    @staticmethod
    def major(root: str) -> Scale:
        return Scale.from_name(root, "major")

    @staticmethod
    def minor(root: str) -> Scale:
        return Scale.from_name(root, "minor")

    @staticmethod
    def dorian(root: str) -> Scale:
        return Scale.from_name(root, "dorian")

    @staticmethod
    def phrygian(root: str) -> Scale:
        return Scale.from_name(root, "phrygian")

    @staticmethod
    def lydian(root: str) -> Scale:
        return Scale.from_name(root, "lydian")

    @staticmethod
    def mixolydian(root: str) -> Scale:
        return Scale.from_name(root, "mixolydian")

    @staticmethod
    def locrian(root: str) -> Scale:
        return Scale.from_name(root, "locrian")

    @staticmethod
    def mode(root: str, mode_name: str) -> Scale:
        return Scale.from_name(root, mode_name)
