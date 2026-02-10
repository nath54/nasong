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
from typing import List, Dict
from dataclasses import dataclass

#
from .pitch import Note, Hz, Pitch
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

        # Major: [0, 2, 4, 5, 7, 9, 11]
        valid_notes = []
        for iv in self.intervals:
            new_pitch = iv.add_to(self.root)
            if isinstance(new_pitch, Note):
                valid_notes.append(new_pitch)
            else:
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
