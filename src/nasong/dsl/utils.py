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


"""Utilities for the NaSong DSL.

This module provides helper functions for higher-level musical operations
within the DSL, such as converting chords into arpeggiated note sequences.
"""

#
### Import Modules. ###
#
from nasong.theory.structures.note import Note
from nasong.theory.structures.chord import Chord
from nasong.theory.core.time import Duration
from nasong.theory.core.pitch import Note as CoreNote


def arpeggiate(
    chord: Chord, pattern: list[int] = [0, 1, 2], duration: Duration = Duration(0.25)
) -> list[Note]:
    """Arpeggiates a chord using a specified index pattern.

    Args:
        chord (Chord): The source chord to arpeggiate.
        pattern (list[int], optional): A list of indices representing the note
            order (e.g., [0, 1, 2] for root, third, fifth). Defaults to [0, 1, 2].
        duration (Duration, optional): The duration of each note in the
            arpeggio. Defaults to a quarter note (0.25).

    Returns:
        list[Note]: A list of `Note` objects forming the arpeggio.
    """
    notes = []
    pitches = chord.pitches
    num_pitches = len(pitches)

    for idx in pattern:
        # Wrap index
        p_idx = idx % num_pitches
        octave_shift = idx // num_pitches

        p = pitches[p_idx]
        if octave_shift != 0:
            if isinstance(p, CoreNote):
                p = p.transpose(octave_shift * 12)
            # handle other pitch types if needed

        notes.append(Note(p, duration=duration, velocity=chord.velocity))

    return notes
