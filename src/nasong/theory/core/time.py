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


"""Time signature and duration representations.

This module provides tools for defining musical meter and note lengths. It
includes classes for `TimeSignature` and `Duration`, along with common
musical constants like `QUARTER` and `EIGHTH`.
"""

#
### Import Modules. ###
#
from dataclasses import dataclass


@dataclass(frozen=True)
class TimeSignature:
    """Represents a musical time signature.

    Attributes:
        numerator (int): Number of beats in a bar.
        denominator (int): The note value that represents one beat.
    """

    numerator: int
    denominator: int

    @property
    def beats_per_bar(self) -> int:
        return self.numerator

    @property
    def beat_value(self) -> int:
        return self.denominator

    def __str__(self):
        return f"{self.numerator}/{self.denominator}"

    def bar_length_in_quarters(self) -> float:
        """Calculates total bar length in quarter notes.

        Returns:
            float: Number of quarters (e.g., 4.0 for 4/4, 3.0 for 6/8).
        """
        return self.numerator * (4 / self.denominator)


@dataclass(frozen=True)
class Duration:
    """Represents a symbolic musical duration.

    Value is expressed in quarter notes (e.g., 1.0 = Quarter, 0.5 = Eighth).

    Attributes:
        value (float): Duration in quarters.
    """

    value: float  # in quarter notes

    def __add__(self, other):
        if isinstance(other, Duration):
            return Duration(self.value + other.value)
        return NotImplemented

    def __mul__(self, other: int | float):
        return Duration(self.value * float(other))


# Common Durations
WHOLE = Duration(4.0)
HALF = Duration(2.0)
QUARTER = Duration(1.0)
EIGHTH = Duration(0.5)
SIXTEENTH = Duration(0.25)
THIRTYSECOND = Duration(0.125)


def dotted(duration: Duration) -> Duration:
    """Applies a dot to a duration (x1.5).

    Args:
        duration (Duration): The base duration.

    Returns:
        Duration: The dotted duration.
    """
    return Duration(duration.value * 1.5)


def triplet(duration: Duration) -> Duration:
    """Applies a triplet modifier to a duration (x2/3).

    Args:
        duration (Duration): The base duration.

    Returns:
        Duration: The triplet duration.
    """
    return Duration(duration.value * (2.0 / 3.0))
