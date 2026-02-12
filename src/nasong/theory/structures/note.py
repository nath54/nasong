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


"""Note event representation.

This module defines the `Note` event class, which combines a pitch with timing
and dynamics for use in rhythmic patterns and sequences.
"""

#
### Import Modules. ###
#
from dataclasses import dataclass

#
from nasong.theory.core.time import Duration, QUARTER
from nasong.theory.core.pitch import Pitch, Note as CoreNote, Hz


@dataclass
class Note:
    """Represents a musical event: a pitch played for a specific duration.

    Attributes:
        pitch (Pitch): The frequency or symbolic note.
        duration (Duration): How long the note lasts. Defaults to QUARTER.
        velocity (float): The strike strength (0.0 to 1.0). Defaults to 1.0.
    """

    pitch: Pitch
    duration: Duration = QUARTER
    velocity: float = 1.0  # 0.0 to 1.0

    def __post_init__(self):
        # Allow passing string to pitch for convenience
        if isinstance(self.pitch, str):
            self.pitch = CoreNote(self.pitch)

    @property
    def name(self) -> str:
        """Returns the string representation of the note's pitch."""
        if isinstance(self.pitch, CoreNote):
            return self.pitch.name
        return str(self.pitch)

    def transpose(self, semitones: int) -> "Note":
        """Creates a new note transposed by the given interval.

        Args:
            semitones (int): The transposition distance.

        Returns:
            Note: A new Note event with the transposed pitch.
        """
        if isinstance(self.pitch, CoreNote):
            return Note(self.pitch.transpose(semitones), self.duration, self.velocity)
        # If Hz, we could transpose frequencies?
        # Pitch doesn't have transpose, Note does.
        # But Hz can be transposed by multiplying frequency.
        # Let's assume Pitch supports or logic to handle it.
        # core.pitch.Hz doesn't have transpose yet but logic exists elsewhere?
        # For now, only CoreNote transpose is safe.
        # If pitch is Hz, we can implement transpose there or here.
        if isinstance(self.pitch, Hz):
            # 2^(semitones/12)
            ratio = 2 ** (semitones / 12.0)
            return Note(Hz(self.pitch.freq * ratio), self.duration, self.velocity)

        return Note(self.pitch, self.duration, self.velocity)

    def with_duration(self, duration: Duration) -> "Note":
        """Returns a copy of the note with a different duration."""
        return Note(self.pitch, duration, self.velocity)
