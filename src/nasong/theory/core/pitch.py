"""
Core Pitch and Frequency Definitions.
"""

from dataclasses import dataclass
from typing import Optional

# math imported but unused removed
from nasong.core.all_values import Value, Constant


# ==========================================
# 1. Tuning Systems
# ==========================================


@dataclass(frozen=True)
class Tuning:
    """
    Defines a tuning system.
    Default is 12-Tone Equal Temperament (12-TET) at A4=440Hz.
    """

    name: str = "12-TET"
    base_freq: float = 440.0
    base_note_index: int = 69  # MIDI index for A4
    divisions: int = 12

    def freq_from_midi(self, midi_index: float) -> float:
        """Calculate frequency from a MIDI index."""
        return self.base_freq * (
            2 ** ((midi_index - self.base_note_index) / self.divisions)
        )

    def freq_from_ratio(self, ratio: float, base_freq: Optional[float] = None) -> float:
        """Calculate frequency from a ratio relative to a base."""
        ref = base_freq if base_freq else self.base_freq
        return ref * ratio


# Global default tuning instance
DEFAULT_TUNING = Tuning()


# ==========================================
# 2. Pitch Representations
# ==========================================


class Pitch:
    """
    Abstract base class for anything that has a frequency.
    """

    def to_hz(self) -> float:
        raise NotImplementedError

    def to_value(self) -> Value:
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
    """
    Explicit raw frequency.
    """

    freq: float

    def to_hz(self) -> float:
        return self.freq

    def __repr__(self):
        return f"<Hz: {self.freq:.2f}>"


@dataclass
class Note(Pitch):
    """
    A symbolic Note (e.g., 'A4', 'C#5').
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
        """
        Parse scientific pitch notation (e.g. 'C4', 'F#5', 'Bb3').
        Returns MIDI index.
        """
        # 1. Normalize name (handle basic flats)
        # Separate letter/accidental from octave
        # Finds the last digit
        import re

        match = re.match(r"([A-Ga-g][b#]?)(-?\d+)", note_str)
        if not match:
            # Fallback: maybe just midi number as string?
            try:
                return int(note_str)
            except ValueError:
                raise ValueError(f"Invalid note format: {note_str}")

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
        return self._midi_index

    @property
    def freq(self) -> float:
        return self.tuning.freq_from_midi(self._midi_index)

    def to_hz(self) -> float:
        return self.freq

    def transpose(self, semitones: int) -> "Note":
        """Return a new Note transposed by semitones."""
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
