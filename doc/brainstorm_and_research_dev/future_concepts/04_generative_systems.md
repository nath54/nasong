# 04_generative_systems.py

```py
# Advanced examples showing "Systemic" or "Generative" music.
# This goes beyond simple loops into rule-based composition.

from nasong.gen import System, Phase
from nasong.theory import Note, Scale
from nasong.instruments import Marimba, Piano

# ==========================================
# Example 1: Phasing (Steve Reich Style)
# ==========================================
# Two identical patterns playing at slightly different speeds.


def piano_phase():
    # The Core Motif
    # E4 F#4 B4 C#5 D5 F#4 E4 C#5 B4 F#4 D5 C#5
    motif = [
        "E4",
        "F#4",
        "B4",
        "C#5",
        "D5",
        "F#4",
        "E4",
        "C#5",
        "B4",
        "F#4",
        "D5",
        "C#5",
    ]

    # Player 1: Steady Tempo
    p1 = Piano().seq(motif, speed=1.0)

    # Player 2: Slightly faster (1.002x speed)
    # They will slowly drift apart and realign in new patterns
    p2 = Piano().seq(motif, speed=1.002)

    return (p1 + p2) * 0.5


# ==========================================
# Example 2: Cellular Automata (Game of Life)
# ==========================================
# Mapping the state of a CA grid to musical parameters.


class GameOfLifeMusic(System):
    def __init__(self, size=8, scale="C major"):
        self.grid = RandomGrid(size, size)
        self.scale = Scale(scale)

    def next_step(self, t):
        """
        Called every 'step' (e.g., every 16th note).
        """
        # Evolve the CA simulation
        self.grid = self.grid.evolve()

        # Map Grid to Music
        # Row 1 -> Kick Drum Activity
        # Row 2 -> Snare Activity
        # Row 3-8 -> Pitch Polyphony

        audio_out = Value(0)

        # Drums
        if self.grid.row(0).any():
            audio_out += Kick()

        # Harmony
        # Active cells in rows 3-8 trigger notes from the scale
        active_indices = self.grid.active_indices(rows=[3, 4, 5, 6, 7])
        for idx in active_indices:
            note = self.scale.degree(idx)  # Map index to scale degree
            audio_out += Marimba().play(note)

        return audio_out


# ==========================================
# Example 3: L-Systems (Fractal Music)
# ==========================================
# String rewriting rules for recursive melody generation.


def fractal_melody():
    # Lindenmayer System
    # Axiom: A
    # Rules: A -> A B, B -> A
    lsys = LSystem(axiom="A", rules={"A": "AB", "B": "A"})

    # Evolve 5 generations: A -> AB -> ABA -> ABAAB ...
    seq_string = lsys.evolve(generations=5)

    # Map symbols to musical actions
    # A = Step Up, B = Step Down
    return Monosynth().interpret_path(
        seq_string,
        start_note="C4",
        actions={
            "A": "+2st",  # Up 2 semitones
            "B": "-1st",  # Down 1 semitone
        },
    )
```