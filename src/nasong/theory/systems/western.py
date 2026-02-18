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


"""Western music theory systems.

This module provides a namespace for Western classical music concepts, including
standard diatonic scales (Major, Minor, Dorian, etc.) and a metaclass for
dynamic Note constant access.
"""

#
### Import Modules. ###
#
from nasong.theory.core.pitch import Note
from nasong.theory.core.scale import Scale


class WesternMeta(type):
    """Metaclass to support dynamic note access.

    Enables syntax like `Western.C4` to return a `Note("C4")` object.
    """

    def __getattr__(cls, name):
        try:
            return Note(name)
        except ValueError as e:
            raise AttributeError(
                f"type object '{cls.__name__}' has no attribute '{name}'"
            ) from e


class Western(metaclass=WesternMeta):
    """Namespace for Western music theory constants and scale factories."""

    # Common Scales factories
    @staticmethod
    def major(root: str) -> Scale:
        """Returns a major scale from the given root."""
        return Scale.from_name(root, "major")

    @staticmethod
    def minor(root: str) -> Scale:
        """Returns a minor scale from the given root."""
        return Scale.from_name(root, "minor")

    @staticmethod
    def dorian(root: str) -> Scale:
        """Returns a dorian scale from the given root."""
        return Scale.from_name(root, "dorian")

    @staticmethod
    def phrygian(root: str) -> Scale:
        """Returns a phrygian scale from the given root."""
        return Scale.from_name(root, "phrygian")

    @staticmethod
    def lydian(root: str) -> Scale:
        """Returns a lydian scale from the given root."""
        return Scale.from_name(root, "lydian")

    @staticmethod
    def mixolydian(root: str) -> Scale:
        """Returns a mixolydian scale from the given root."""
        return Scale.from_name(root, "mixolydian")

    @staticmethod
    def locrian(root: str) -> Scale:
        """Returns a locrian scale from the given root."""
        return Scale.from_name(root, "locrian")

    @staticmethod
    def mode(root: str, mode_name: str) -> Scale:
        """Returns a mode scale from the given root and mode name."""
        return Scale.from_name(root, mode_name)
