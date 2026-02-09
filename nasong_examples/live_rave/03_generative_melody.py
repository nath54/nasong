import random
from nasong.core.all_values import Value, Constant, Triangle, ADSR2, Sin
from nasong.theory import render
from nasong.theory.structures.progression import Progression
from nasong.theory.structures.chord import Chord
from nasong.theory.core.time import QUARTER, EIGHTH, SIXTEENTH
from nasong.theory.systems.western import Western
from nasong.theory.systems.east_asian import EastAsian

# === Generative Logic ===


def generate_progression(length=16):
    # Pick a random scale
    root = random.choice(["C4", "D4", "E4", "F4", "G4", "A4"])
    sys_type = random.choice(["western", "yo"])

    if sys_type == "western":
        scale = Western.minor(root)
    else:
        scale = EastAsian.yo_scale(root)

    print(f"Generated Scale: {scale.name} on {root}")

    # Generate random chords/notes
    chords = []
    durations = []

    # Simple random walk melody
    current_index = 0

    for _ in range(length):
        # Move up or down by small intervals
        step = random.choice([-2, -1, 0, 1, 2, 3])
        current_index += step

        # Keep within reasonable bounds (manual clamping around scale center)
        # Note: Scale doesn't have bounds, but array access does if we wrap or not.
        # Let's just use raw indices and map them to pitches if Scale supports it?
        # Scale.degree supports arbitrary index (0 = root).

        # We need a Chord object for Progression
        # Single note chord
        pitch = scale.degree(current_index + 1)  # degree is 1-based
        chord = Chord(root=pitch, intervals=[], name="gen_note")
        chords.append(chord)

        # Random duration
        dur = random.choice([QUARTER, EIGHTH, EIGHTH])
        durations.append(dur)

    return Progression(chords, durations)


# === Instrument ===


def plucky_synth(time, freq, start, duration, velocity):
    # Short pluck
    osc = Triangle(time, freq)
    # FM modulation for texture
    mod = Sin(time, freq * 2.0) * 500.0 * ExponentialDecay(time, start, 0.1)
    osc_mod = Triangle(time, freq + mod)

    env = ADSR2(time, start, duration, 0.01, 0.2, 0.0, 0.0)
    return osc_mod * env * 0.5 * velocity


# === Build ===

prog = generate_progression(32)  # 32 notes
sequencer = render(prog, time_value=None, instrument_factory=plucky_synth, bpm=140)
