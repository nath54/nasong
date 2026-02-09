import numpy as np
from nasong.theory.systems.western import Western
from nasong.theory.structures.progression import Progression
from nasong.theory.core.time import HALF
from nasong.theory import render
from nasong.core.all_values import Constant, Sawtooth, ADSR2
from nasong.core.wav import WavUtils
from scipy.io import wavfile


def synth_wrapper(time, freq, start, duration, velocity):
    # freq: float (from Constant or raw float? render passes float, so here it is float)
    # start: float
    # duration: float
    # velocity: float

    # Oscillator (Continuous phase tracking is hard in pure functional if freq changes per note)
    # But render creates new Synth for each note event.
    # So 'time' is global time.

    # Simple Sawtooth
    # We use 'value=time' for the Sawtooth input phase driver
    osc = Sawtooth(value=time, frequency=Constant(freq))

    # Envelope
    env = ADSR2(
        time=time,
        note_start=start,
        note_duration=duration,
        attack_time=0.01,
        decay_time=0.1,
        sustain_level=0.7,
        release_time=0.2,
    )

    return osc * env * Constant(velocity * 0.5)


def run_demo():
    print("Generating Theory Demo...")

    # 1. Define Scale
    scale = Western.major("C4")

    # 2. Define Progression
    # I - V - vi - IV
    prog = Progression.from_roman_numerals(scale, ["I", "V", "vi", "IV"], duration=HALF)
    print(f"Progression: {[c.name for c in prog.chords]}")

    # 3. Render to Sequencer
    # mapped to our synth wrapper
    seq = render(prog, time_value=None, instrument_factory=synth_wrapper, bpm=120)

    # 4. Generate Audio
    sr = 44100
    duration = 5.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    print(f"Generating {duration:.2f}s of audio...")
    audio = seq.getitem_np(t, sr)

    # 5. Save
    fname = "demo_full_pipeline.wav"
    audio_int16 = WavUtils.prepare_signal(audio)
    wavfile.write(fname, sr, audio_int16)
    print(f"✅ Successfully saved audio to '{fname}'")


if __name__ == "__main__":
    run_demo()
