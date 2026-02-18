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


"""Pitch and tuning system representations.

This module provides the core abstractions for musical pitch, including raw
frequencies (Hz), symbolic notes (Pitch Class + Octave), and customizable
tuning systems (defaults to 12-TET A4=440Hz).
"""

#
### Import Modules. ###
#
from typing import Optional
from dataclasses import dataclass

#
import re

#
from nasong.core.value import Value
from nasong.core.values.basic.value_constant import Constant


# ==========================================
# 1. Tuning Systems
# ==========================================


@dataclass(frozen=True)
class Tuning:
    """Defines a musical tuning system.

    Attributes:
        name (str): Identifier for the tuning (e.g., "12-TET").
        base_freq (float): Frequency of the reference note.
        base_note_index (int): MIDI index of the reference note.
        divisions (int): Number of steps per octave.
    """

    name: str = "12-TET"
    base_freq: float = 440.0
    base_note_index: int = 69  # MIDI index for A4
    divisions: int = 12

    def freq_from_midi(self, midi_index: float) -> float:
        """Calculates the absolute frequency for a given MIDI index.

        Args:
            midi_index (float): The potentially fractional MIDI note number.

        Returns:
            float: The frequency in Hz.
        """
        return self.base_freq * (
            2 ** ((midi_index - self.base_note_index) / self.divisions)
        )

    def freq_from_ratio(self, ratio: float, base_freq: Optional[float] = None) -> float:
        """Calculates a frequency from a ratio relative to a reference.

        Args:
            ratio (float): The frequency multiplier.
            base_freq (float, optional): The reference frequency. Defaults to
                the system's base_freq.

        Returns:
            float: The resulting frequency in Hz.
        """
        ref = base_freq if base_freq else self.base_freq
        return ref * ratio


# Global default tuning instance
DEFAULT_TUNING = Tuning()


# ==========================================
# 2. Pitch Representations
# ==========================================


class Pitch:
    """Abstract base class for all frequency-carrying objects."""

    def to_hz(self) -> float:
        """Converts the pitch representation to raw Hz.

        Returns:
            float: Frequency in Hz.
        """
        raise NotImplementedError

    def to_value(self) -> Value:
        """Wraps the frequency into a NaSong Constant value.

        Returns:
            Value: A constant audio graph node.
        """
        return Constant(self.to_hz())

    def __float__(self):
        return self.to_hz()

    def __add__(self, other):
        # Allow adding intervals (in semitones) or Hz?
        # For now, let's assume adding pitch + pitch is not defined,
        # but pitch + interval is.
        raise NotImplementedError


@dataclass
class Hz(Pitch):
    """Represents an explicit frequency value in Hertz.

    Attributes:
        freq (float): The raw frequency value.
    """

    freq: float

    def to_hz(self) -> float:
        """Returns the stored frequency."""
        return self.freq

    def __repr__(self):
        return f"<Hz: {self.freq:.2f}>"


@dataclass
class Note(Pitch):
    """Represents a symbolic musical note (e.g., 'C4', 'A#5').

    Supports scientific pitch notation and automatic conversion to MIDI or Hz
    using a provided Tuning system.

    Attributes:
        name (str): The note name (e.g., 'Db3').
        tuning (Tuning): The tuning system to use. Defaults to 12-TET.
    """

    name: str
    tuning: Tuning = DEFAULT_TUNING

    # Note name parsing utils
    _NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    _NOTE_MAP = {name: i for i, name in enumerate(_NOTE_NAMES)}

    # Enharmonic equivalents
    _ENHARMONICS = {
        "Db": "C#",
        "eb": "D#",
        "Gb": "F#",
        "Ab": "G#",
        "Bb": "A#",
        "Eb": "D#",  # Common ones
    }

    def __post_init__(self):
        self._midi_index = self._parse_note(self.name)

    def _parse_note(self, note_str: str) -> int:
        """Parses scientific pitch notation into a MIDI index.

        Args:
            note_str (str): The note identifier (e.g., 'F#2').

        Returns:
            int: Corresponding MIDI note number.

        Raises:
            ValueError: If the string format is invalid or unknown.
        """
        # 1. Normalize name (handle basic flats)
        # Separate letter/accidental from octave
        # Finds the last digit

        match = re.match(r"([A-Ga-g][b#]?)(-?\d+)", note_str)
        if not match:
            # Fallback: maybe just midi number as string?
            try:
                return int(note_str)
            except ValueError as e:
                raise ValueError(f"Invalid note format: {note_str}") from e

        note_part, octave_part = match.groups()

        # Normalize casing
        note_part = note_part.capitalize()
        if note_part in self._ENHARMONICS:
            note_part = self._ENHARMONICS[note_part]

        if note_part not in self._NOTE_MAP:
            # Handle double sharps/flats manually if needed later
            raise ValueError(f"Unknown note name: {note_part}")

        semitone = self._NOTE_MAP[note_part]
        octave = int(octave_part)

        # MIDI note calculation: C_neg1 is 0. C0 is 12. C4 is 60.
        # Formula: (octave + 1) * 12 + semitone
        midi = (octave + 1) * 12 + semitone
        return midi

    @property
    def midi(self) -> int:
        """The MIDI note number."""
        return self._midi_index

    @property
    def freq(self) -> float:
        """The frequency in Hz."""
        return self.tuning.freq_from_midi(self._midi_index)

    def to_hz(self) -> float:
        """The frequency in Hz."""
        return self.freq

    def transpose(self, semitones: int) -> "Note":
        """Transposes the note by a fixed number of semitones.

        Args:
            semitones (int): Steps to move (positive or negative).

        Returns:
            Note: A new Note object representing the transposed pitch.
        """
        new_midi = self._midi_index + semitones
        # We need to reverse map midi to string if we want to keep it as Note
        # But this is lossy (C# vs Db).
        # For now, let's reconstruct a standard name.
        octave = (new_midi // 12) - 1
        semi = new_midi % 12
        new_name = f"{self._NOTE_NAMES[semi]}{octave}"
        return Note(new_name, self.tuning)

    def __repr__(self):
        return f"<Note {self.name} ({self.midi})>"
