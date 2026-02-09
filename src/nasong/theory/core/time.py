"""
Core Time and Rhythmic Definitions.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Union


@dataclass(frozen=True)
class TimeSignature:
    """
    Represents a time signature (e.g. 4/4, 6/8).
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
        """
        Returns the length of a bar in quarter notes.
        e.g. 4/4 -> 4.0
             3/4 -> 3.0
             6/8 -> 6 * (4/8) = 3.0
        """
        return self.numerator * (4 / self.denominator)


@dataclass(frozen=True)
class Duration:
    """
    Represents a musical duration in terms of quarter notes.
    1.0 = Quarter Note
    0.5 = Eighth Note
    4.0 = Whole Note
    """

    value: float  # in quarter notes

    def __add__(self, other):
        if isinstance(other, Duration):
            return Duration(self.value + other.value)
        return NotImplemented

    def __mul__(self, other: Union[int, float]):
        return Duration(self.value * float(other))


# Common Durations
WHOLE = Duration(4.0)
HALF = Duration(2.0)
QUARTER = Duration(1.0)
EIGHTH = Duration(0.5)
SIXTEENTH = Duration(0.25)
THIRTYSECOND = Duration(0.125)


def dotted(duration: Duration) -> Duration:
    """Return a dotted version of the duration (1.5x)"""
    return Duration(duration.value * 1.5)


def triplet(duration: Duration) -> Duration:
    """Return a triplet version of the duration (2/3x)"""
    return Duration(duration.value * (2.0 / 3.0))
