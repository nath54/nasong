# 02_instrument_dsl.py

```py
# This file demonstrates the "NaSL" (NaSong Language) Internal DSL.
# The goal is to make signal graphs (patching) readable and concise.

from nasong.dsl import instrument, unit, chain
from nasong.dsl.modules import Osc, Env, Filter, Amp, Eff
from nasong.dsl.units import Hz, Ms, dB

# ==========================================
# 1. The @instrument Decorator
# ==========================================
# This decorator handles:
# - Wraps scalar inputs into Constant Values.
# - Compiles the signal graph at initialization (not render time).
# - Exposes parameters for automation.


@instrument
def SimpleSubSynth(freq, gate):
    """
    A simple subtractive synthesizer voice.
    Input: freq (Hz), gate (0/1 trigger)
    """

    # Envelopes
    # Syntax: Env.ADSR(gate, attack, decay, sustain, release)
    amp_env = Env.ADSR(gate, 5 * Ms, 200 * Ms, 0.7, 300 * Ms)
    filt_env = Env.ADSR(gate, 20 * Ms, 300 * Ms, 0.2, 500 * Ms)

    # Oscillators
    # We can mix oscillators purely with math operators
    # Note: 'freq' is automatically a Value here
    osc1 = Osc.Saw(freq)
    osc2 = Osc.Square(freq * 0.5) * 0.5  # Sub-oscillator (-1 octave, half vol)

    source = osc1 + osc2

    # Filter Modulation
    # Map envelope (0-1) to frequency range (100Hz - 5000Hz)
    cutoff = filt_env.map(100, 5000)

    # Signal Chain utilizing the `chain` or `>>` syntax
    # Source -> LowPass Filter -> VCA (Amp)
    output = source >> Filter.LowPass(cutoff, res=0.3) >> Amp(amp_env)

    return output


# ==========================================
# 2. Advanced FM Synthesis Example
# ==========================================


@instrument
def GlassyFM(freq, gate, ratio=2.0, index=500):
    """
    2-Operator FM Synth.
    Arguments 'ratio' and 'index' are now controllable parameters
    that can be automated!
    """

    # Modulator
    mod_index_env = Env.AR(gate, 10 * Ms, 1000 * Ms)
    mod_freq = freq * ratio
    modulator = Osc.Sine(mod_freq) * index * mod_index_env

    # Carrier
    # The modulator output is added to the carrier's frequency
    carrier = Osc.Sine(freq + modulator)

    return carrier * Env.AR(gate, 5 * Ms, 2000 * Ms)


# ==========================================
# 3. Usage: Sequencing instruments
# ==========================================


def song():
    # Instantiate the defined instrument
    # We can pass static values OR automation curves
    bass = SimpleSubSynth(
        freq=Sequence([55, 55, 110, 55], duration=0.25),
        gate=Sequence([1, 1, 1, 1], duration=0.25),
    )

    # Lead with automation on the "ratio" parameter
    lead_line = GlassyFM(
        freq=Sequence([440, 660, 880], duration=0.5),
        gate=Pulse(0.5),
        ratio=Line(start=1.0, end=4.0, duration=10),  # Automation!
    )

    return Mix([bass, lead_line])
```