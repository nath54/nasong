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


"""Musical interval representations.

This module defines the `Interval` class, which handles musical distance using
either semitones (12-Tone Equal Temperament) or frequency ratios (Just
Intonation). It provides utilities for transposing pitches and parsing
interval names.
"""

#
### Import Modules. ###
#
from dataclasses import dataclass

#
from .pitch import Pitch, Hz, Note


@dataclass
class Interval:
    """Represents a musical interval.

    An interval can be defined by semitones (logarithmic scale) or a frequency
    ratio (linear scale).

    Attributes:
        semitones (float): The distance in semitones.
        ratio (float): The corresponding frequency ratio (e.g., 2.0 for octave).
    """

    semitones: float = 0.0
    ratio: float = 1.0

    def __init__(self, value: int | float | str) -> None:
        """Initializes the Interval.

        Args:
            value (int | float | str):
                - If int/float: The number of semitones (e.g., 7.0 for a P5).
                - If str: The abbreviated name of the interval (e.g., "P5", "m3").
        """
        # Simple implementation for now: everything is semitones
        if isinstance(value, (int, float)):
            self.semitones = float(value)
            self.ratio = 2 ** (self.semitones / 12.0)
        elif isinstance(value, str):
            target = self._parse_name(value)
            self.semitones = float(target)
            self.ratio = 2 ** (self.semitones / 12.0)

    def _parse_name(self, name: str) -> int:
        """Parses a symbolic interval name into a semitone count.

        Args:
            name (str): The name (e.g., "M3", "tritone").

        Returns:
            int: Number of semitones.

        Raises:
            ValueError: If the name is unknown.
        """
        # Basic lookup for common names
        lookup = {
            "P1": 0,
            "unison": 0,
            "m2": 1,
            "min2": 1,
            "S": 1,  # Minor 2nd / Semitone
            "M2": 2,
            "maj2": 2,
            "T": 2,  # Major 2nd / Tone
            "m3": 3,
            "min3": 3,
            "M3": 4,
            "maj3": 4,
            "P4": 5,
            "perf4": 5,
            "TT": 6,
            "tritone": 6,
            "d5": 6,
            "A4": 6,
            "P5": 7,
            "perf5": 7,
            "m6": 8,
            "min6": 8,
            "M6": 9,
            "maj6": 9,
            "m7": 10,
            "min7": 10,
            "M7": 11,
            "maj7": 11,
            "P8": 12,
            "octave": 12,
        }
        if name in lookup:
            return lookup[name]
        # Try to handle negatives? "-P5"
        raise ValueError(f"Unknown interval name: {name}")

    def add_to(self, pitch: Pitch) -> Pitch:
        """Applies this interval to a Pitch object (transposition).

        Args:
            pitch (Pitch): The starting pitch (Hz or Note).

        Returns:
            Pitch: The resulting transposed pitch.

        Raises:
            TypeError: If the input is not a recognized Pitch type.
        """
        if isinstance(pitch, Note):
            # Transpose by semitones
            # Note: Note.transpose expects int currently, but we might have float semitones (microtonal)
            # If float, we might need to return Hz or a DetunedNote
            if self.semitones.is_integer():
                return pitch.transpose(int(self.semitones))
            else:
                # Microtonal transposition -> Return Hz
                return Hz(pitch.freq * self.ratio)

        elif isinstance(pitch, Hz):
            return Hz(pitch.freq * self.ratio)

        raise TypeError(f"Cannot add interval to {type(pitch)}")

    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self.semitones + other.semitones)
        return NotImplemented

    def __neg__(self) -> "Interval":
        """Returns the inverse interval (e.g., -P5)."""
        return Interval(-self.semitones)

    def __repr__(self):
        return f"<Interval: {self.semitones}st ({self.ratio:.2f}x)>"
