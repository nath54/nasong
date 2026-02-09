"""
Chord Structure.
"""

from dataclasses import dataclass, field
from typing import List, Union, Optional
from nasong.theory.core.pitch import Pitch, Note as CoreNote, Hz
from nasong.theory.core.interval import Interval
from nasong.theory.core.time import Duration, QUARTER
from nasong.theory.structures.note import Note


@dataclass
class Chord:
    """
    Represents a simultaneous sounding of multiple pitches.
    """

    root: Pitch
    intervals: List[Interval]
    duration: Duration = QUARTER
    velocity: float = 1.0
    name: str = "custom"
    inversion: int = 0

    def __post_init__(self):
        if isinstance(self.root, str):
            self.root = CoreNote(self.root)

    @property
    def pitches(self) -> List[Pitch]:
        """
        Returns the pitches in the chord, accounting for inversion.
        """
        raw_pitches = [self.root]
        for iv in self.intervals:
            # interval logic...
            # if intervals are from root
            p = iv.add_to(self.root)
            raw_pitches.append(p)

        # Handle inversion
        if self.inversion > 0:
            # Move bottom notes up an octave
            # This is a naive implementation
            for _ in range(self.inversion):
                p = raw_pitches.pop(0)
                if isinstance(p, CoreNote):
                    p = p.transpose(12)
                elif isinstance(p, Hz):
                    p = Hz(p.freq * 2.0)
                raw_pitches.append(p)

        return raw_pitches

    @property
    def notes(self) -> List[Note]:
        """
        Returns a list of Note events for this chord.
        """
        return [Note(p, self.duration, self.velocity) for p in self.pitches]

    @classmethod
    def from_name(
        cls, root: str, quality: str, duration: Duration = QUARTER
    ) -> "Chord":
        """
        Factory for common chord qualities.
        """
        root_note = CoreNote(root)
        intervals = _CHORD_QUALITY_INTERVALS.get(quality.lower())
        if not intervals:
            raise ValueError(f"Unknown chord quality: {quality}")

        iv_objs = [Interval(s) for s in intervals]
        return cls(root_note, iv_objs, duration=duration, name=quality)


# Common Interval Patterns (semitones from root)
# Note: Root (0) is implicit in the class logic above if we add intervals to root
# But usually Chord definitions list intervals *above* root.
# So Major Triad = [4, 7] (Major 3rd, Perfect 5th)
# The root is always index 0 of pitches list.

_CHORD_QUALITY_INTERVALS = {
    "major": [4, 7],
    "maj": [4, 7],
    "M": [4, 7],
    "minor": [3, 7],
    "min": [3, 7],
    "m": [3, 7],
    "dim": [3, 6],
    "aug": [4, 8],
    "sus2": [2, 7],
    "sus4": [5, 7],
    "maj7": [4, 7, 11],
    "min7": [3, 7, 10],
    "dom7": [4, 7, 10],
    "7": [4, 7, 10],
    "m7b5": [3, 6, 10],  # Half-diminished
    "dim7": [3, 6, 9],  # Fully diminished
}
