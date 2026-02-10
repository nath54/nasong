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
from typing import List

#
from nasong.theory.structures.note import Note
from nasong.theory.structures.chord import Chord
from nasong.theory.core.time import Duration
from nasong.theory.core.pitch import Note as CoreNote


def arpeggiate(
    chord: Chord, pattern: List[int] = [0, 1, 2], duration: Duration = Duration(0.25)
) -> List[Note]:
    """
    Arpeggiate a chord using a pattern of indices.
    [0, 1, 2] -> Root, 3rd, 5th (if triad).
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
