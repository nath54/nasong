"""
Core Scale Definitions.
"""

from dataclasses import dataclass, field
from typing import List, Union, Dict
from .pitch import Note, Hz, Pitch, Tuning, DEFAULT_TUNING
from .interval import Interval


@dataclass
class Scale:
    """
    Represents a musical scale built from a root note and a pattern of intervals.
    """

    root: Note
    intervals: List[Interval]
    name: str = "custom"

    def __post_init__(self):
        # Cache the notes
        self._notes = self._generate_notes()

    def _generate_notes(self) -> List[Note]:
        """
        Generate the notes of the scale relative to the root.
        """
        notes = []
        for interval in self.intervals:
            # Add interval to root
            # Interval.add_to(Note) -> Note
            # We assume Interval(0) is the first element for the root itself
            # If not present, we should add it?
            # Convention: intervals list usually includes 0 or we act as if it does.
            # Let's enforce 0 being present or add root manually.
            pass

        # New approach: Interval list defines steps FROM ROOT.
        # e.g. Major: [0, 2, 4, 5, 7, 9, 11]
        valid_notes = []
        for iv in self.intervals:
            new_pitch = iv.add_to(self.root)
            if isinstance(new_pitch, Note):
                valid_notes.append(new_pitch)
            else:
                # Handle microtonal resulting pitch?
                # For now, simplistic implementation keeps it if it's a Note.
                # If we get Hz, we might need a "DetunedNote" or just keep Hz.
                # But 'notes' property implies Note objects.
                # Let's store generalized Pitch objects.
                valid_notes.append(new_pitch)
        return valid_notes

    @property
    def notes(self) -> List[Pitch]:
        return self._notes

    def degree(self, index: int) -> Pitch:
        """
        Get the note at the given scale degree (1-based index).
        Handles wrapping (octaves).
        """
        # 1-based index to 0-based
        idx = index - 1
        num_notes = len(self._notes)

        # Calculate octave shift
        octave_shift = idx // num_notes
        local_idx = idx % num_notes

        base_note = self._notes[local_idx]

        # Transpose base_note by octave_shift * 12 semitones?
        # This assumes 12-TET. For generic scales, we should use the 'period' (last interval?)
        # For standard scales, the period is an octave (12 semitones).
        # Let's assume octave period for now.

        if isinstance(base_note, Note):
            return base_note.transpose(octave_shift * 12)
        elif isinstance(base_note, Hz):
            return Hz(base_note.freq * (2**octave_shift))

        raise TypeError("Unknown pitch type in scale")

    @classmethod
    def from_name(cls, root_name: str, scale_name: str) -> "Scale":
        """
        Factory to create a scale from a root name and a scale name (e.g. "C", "major").
        """
        root = Note(root_name)
        intervals = _SCALE_PATTERNS.get(scale_name.lower())
        if not intervals:
            raise ValueError(f"Unknown scale name: {scale_name}")

        # Convert integers to Intervals
        iv_objs = [Interval(s) for s in intervals]
        return cls(root, iv_objs, name=scale_name)


# Predefined Scale Patterns (Semitones from root)
_SCALE_PATTERNS: Dict[str, List[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],  # Natural minor
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],  # Ascending
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],  # Hexatonic blues
    "chromatic": list(range(12)),
    "whole_tone": [0, 2, 4, 6, 8, 10],
    "diminished_wh": [0, 2, 3, 5, 6, 8, 9, 11],  # Whole-Half
    "diminished_hw": [0, 1, 3, 4, 6, 7, 9, 10],  # Half-Whole
}
