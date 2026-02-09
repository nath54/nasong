from nasong.core.all_values import (
    Value,
    Constant,
    Sin,
    ADSR2,
    WhiteNoise,
)
from nasong.theory import render
from nasong.theory.structures.progression import Progression
from nasong.theory.core.time import QUARTER
from nasong.theory.systems.western import Western

# === 1. Define Instruments ===


def kick_drum(
    time: Value, freq: Value, start: float, duration: float, velocity: float
) -> Value:
    # Pitch envelope: Rapid drop from high to low
    pitch_decay = ADSR2(
        time,
        start,
        duration,
        attack_time=0.001,
        decay_time=0.05,
        sustain_level=0.0,
        release_time=0.1,
    )
    # Base freq is 50Hz, pitch decay adds 150Hz sweep
    f = Constant(50.0) + (pitch_decay * 150.0)

    osc = Sin(time, f)

    # Amplitude Envelope
    amp = ADSR2(
        time,
        start,
        duration,
        attack_time=0.001,
        decay_time=0.3,
        sustain_level=0.0,
        release_time=0.1,
    )

    # Click for transient
    click = WhiteNoise() * ADSR2(time, start, 0.01, 0.0, 0.01, 0.0, 0.0) * 0.5

    return (osc * amp * 0.8) + click


def bass_synth(
    time: Value, freq: Value, start: float, duration: float, velocity: float
) -> Value:
    # Simple offbeat bass
    osc = Sin(time, freq)
    amp = ADSR2(time, start, duration, 0.01, 0.1, 0.6, 0.1)
    return osc * amp * 0.6


# === 2. Define Composition ===


# 4/4 Basic Beat: Kick on 1, 2, 3, 4
def make_beat():
    scale = Western.major("C2")
    # 4 kicks
    kicks = Progression.from_roman_numerals(
        scale, ["I", "I", "I", "I"], duration=QUARTER
    )

    # Offbeat bass: on the "and" of each beat
    # We can overlay progressions by summing sequencers later!
    # Or create a complex rhythm.

    return kicks


# === 3. Render ===

# Create Kicks
prog_kick = make_beat()  # 1 Bar
# Loop it manually by repeating chords?
# For live coding, we usually want infinite loop.
# Current render() creates a finite Sequencer.
# Workaround: Create a long progression
long_prog = Progression(prog_kick.chords * 16, [QUARTER] * (4 * 16))

sequencer = render(long_prog, time_value=None, instrument_factory=kick_drum, bpm=130)

# To add HiHats/Bass, we would create another sequencer and sum them:
# sequencer = seq_kick + seq_bass
