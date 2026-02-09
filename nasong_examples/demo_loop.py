from nasong.theory.systems.western import Western
from nasong.theory.structures.progression import Progression
from nasong.theory import render
from nasong.core.values.basic.value_identity import Identity
from nasong.core.values.complex.value_sin import Sin
from nasong.core.values.complex.value_adsr2 import ADSR2
from nasong.core.values.complex.value_square import Square
from nasong.core.values.complex.value_sawtooth import Sawtooth
from nasong.core.values.single_itms_ops.value_low_pass import LowPass

# Configuration
BPM = 120
# Duration of a 4-bar loop (16 quarters)
# (16 quarters / BPM) * 60 seconds
LOOP_QUARTERS = 4
LOOP_SECONDS = (LOOP_QUARTERS * 60.0) / BPM

# 1. Define Looping Time
# This is the secret for infinite loops!
time = Identity() % LOOP_SECONDS


# 2. Define Instruments
def acid_bass(time, freq, start, dur, vel):
    """A gritty bass with LFO resonant filter."""
    # Envelopes
    env = ADSR2(time, start, dur, 0.05, 0.1, 0.8, 0.5)

    # Osc: Saw + Square for thickness
    osc = (Sawtooth(time, freq) + Square(time, freq)) * 0.5

    # Wobble LFO (modulating filter cutoff)
    wobble = (Sin(time, 2.0) + 1.0) * 0.5  # 0 to 1
    cutoff = 300 + (wobble * 2000 * env)  # Env also affects cutoff

    res_bass = LowPass(osc, cutoff)
    return res_bass * env * vel * 0.3


def space_pad(time, freq, start, dur, vel):
    """A slow-attack pad with subtle vibrato."""
    # Slow attack for pads
    env = ADSR2(time, start, dur, 0.5, 0.5, 0.8, 0.5)

    # Vibrato LFO
    vibrato = Sin(time, 6.0) * 2.0  # 2Hz vibrato
    osc = Sin(time, freq + vibrato)

    return osc * env * vel * 0.2


# 3. Composition
# A cool minor progression
bass_prog = Progression([Western.D2, Western.Eb2, Western.G2, Western.Bb1])
pad_prog = Progression([Western.C4, Western.Eb4, Western.G4, Western.Bb3])

# 4. Render to Sequencer
# We can sum multiple sequencers!
bass_layer = render(bass_prog, time, acid_bass, bpm=BPM)
pad_layer = render(pad_prog, time, space_pad, bpm=BPM)

# The final 'sequencer' variable is what the App looks for
sequencer = bass_layer + pad_layer
