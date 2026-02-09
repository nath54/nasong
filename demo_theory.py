import numpy as np
from nasong.theory.systems.western import Western
from nasong.theory.structures.progression import Progression
from nasong.theory.core.time import QUARTER, HALF
from nasong.theory import render

from nasong.core.all_values import Value, Sin, Constant, ADSR2, Product, Sawtooth
from nasong.core.all_values import Identity
from nasong.core.wav import WavUtils

# 1. Define Progression
# C Major: I - V - vi - IV (Pop progression)
scale = Western.major("C4")
prog = Progression.from_roman_numerals(scale, ["I", "V", "vi", "IV"], duration=HALF)

# 2. Define Time Value (global time)
time_val = Identity()


# 3. Define Instrument Factory
def synth_wrapper(time, freq, start, duration, velocity):
    # freq: float
    # start: float
    # duration: float (seconds)
    # velocity: float

    # Oscillator (Continuous phase)
    # Sawtooth takes 'value' as the phase input (usually time)
    # and 'frequency' as Value
    osc = Sawtooth(value=time, frequency=Constant(freq))

    # Envelope
    # ADSR2 takes time, note_start, note_duration
    env = ADSR2(
        time=time,
        note_start=start,
        note_duration=duration,
        attack_time=0.01,
        decay_time=0.1,
        sustain_level=0.5,
        release_time=0.2,
    )

    # Amp
    # Velocity scaling
    amp = Product(
        [osc, env, Constant(velocity * 0.1)]
    )  # Lower volume to avoid clipping
    return amp


# 4. Render to Sequencer
# render returns a Sequencer instance
seq = render(prog, time_val, instrument_factory=synth_wrapper, bpm=120.0)

# 5. Generate Audio
sample_rate = 44100
total_duration = prog.duration.value * (60.0 / 120.0) + 1.0  # +1 sec tail
num_samples = int(total_duration * sample_rate)

# Generate time array (seconds)
time_array = np.linspace(0, total_duration, num_samples, dtype=np.float32)

print(f"Generating {total_duration:.2f}s of audio...")

# Sequencer.getitem_np expects indexes_buffer to be input to time_val.
# Since time_val is Identity(), indexes_buffer IS the time.
audio = seq.getitem_np(time_array, sample_rate)

# 6. Save WAV
WavUtils.save_wav_file("demo_theory.wav", sample_rate, WavUtils.prepare_signal(audio))
