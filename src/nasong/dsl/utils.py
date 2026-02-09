"""
DSL Utilities.
"""

from nasong.theory.structures.note import Note
from nasong.theory.structures.chord import Chord
from nasong.theory.core.time import Duration
from nasong.theory.core.pitch import Pitch, Note as CoreNote
from typing import List, Union


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
