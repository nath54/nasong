from nasong.theory.core.scale import Scale
from nasong.theory.structures.progression import Progression
from nasong.theory.systems.east_asian import EastAsian
from nasong.theory.core.time import QUARTER, EIGHTH
from typing import List
import random


class Koto:
    """
    Generates Koto-style melodic patterns.
    """

    @staticmethod
    def traditional_motif(root: str = "D4"):
        """
        Generates a motif using the In scale.
        """
        scale = EastAsian.in_scale(root)

        # Koto music often uses arpeggiated clusters or specific intervals (4ths, 5ths)
        # Not typically chord progressions in the Western sense.
        # Returning a sequence of notes as a single-chord "Progression" for now?
        # Or a "Melody"?
        # DSL structure needs to support Melodies better.
        # For now, using Progression as a container of chords where each chord is a single note (monophonic)

        # Example pattern: Root -> 5th -> 4th -> Root
        # Indices: 0, 3, 2, 0
        from nasong.theory.structures.chord import Chord

        if hasattr(scale, "notes"):
            # Indices modulo length
            notes = [scale.notes[i % len(scale.notes)] for i in [0, 3, 2, 0]]
        else:
            # Fallback
            notes = []
        # Create single-note chords
        chords = [Chord(root=n, intervals=[], name="Note") for n in notes]

        return Progression(chords, [QUARTER] * 4)
