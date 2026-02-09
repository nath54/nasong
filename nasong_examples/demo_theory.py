from nasong.core.all_values import Sin, Constant, ADSR2
from nasong.theory import render
from nasong.theory.structures.progression import Progression
from nasong.theory.systems.western import Western


# Minimal synth
def beep(time, freq, start, dur, vel):
    # ADSR2(time, note_start, note_duration, attack, decay, sustain, release)
    env = ADSR2(time, start, dur, 0.01, 0.1, 0.5, 0.1)
    return Sin(time, freq) * env * vel * 0.5


# Minimal pattern
prog = Progression([Western.C4, Western.E4, Western.G4, Western.C5])
sequencer = render(prog, time_value=None, instrument_factory=beep, bpm=120)
