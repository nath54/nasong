# 03_live_session.py

```py
# This simulates the structure of a live-code file during an "Algo-Rave".
# The user edits this file, saves it, and the engine hot-swaps the audio generation.

from nasong.live import Session, Clock, Track
from nasong.theory import Scale, Progression
from nasong.dsl.patterns import Euclid, StringPat
from nasong.instruments import Kick909, HatClosed, SimpleSubSynth

# ==========================================
# Global Session Context
# ==========================================
# This defines the "transport" (clock) and harmonic context.
# Changing 'bpm' here immediately updates the master clock.

session = Session(bpm=135, key="F", scale="minor")

# ==========================================
# Tracks / Loops
# ==========================================
# Each 'Track' runs a generator function locked to the session clock.


@session.track("drums")
def drums(t):
    """
    Drum pattern using string-based sequencing.
    """
    # Kick: 4-on-the-floor
    # "x" = trigger, "-" = rest, "." = 1/2 step (16th note usually)
    kick = Kick909().trig("x---x---x---x---")

    # Hats: 16th note pattern
    # We can inject randomness for humanization
    hats = HatClosed().trig("x-x-x-x-xxx-x-x-").humanize(0.1)

    return kick + hats * 0.5


@session.track("bass")
def bassline(t):
    """
    Generative bassline.
    Uses the session's key context implicitly or explicitly.
    """
    # Euclidean Rhythm: 5 hits distributed over 16 steps
    rhythm = Euclid(16, 5)

    # Harmonic progression
    prob = Progression("i VI VII v")

    # The synth
    # .notes() automatically maps scale degrees/chords to frequencies
    return SimpleSubSynth(decay=0.2).play(notes=prob, rhythm=rhythm, octave=2)


@session.track("pads")
def atmosphere(t):
    """
    Slow evolving texture.
    Stateful: 'Phaser' relies on continuous time 't'.
    """
    # Slow LFO for filter cutoff
    cutoff = SineLFO(freq=0.1).map(400, 2000)

    # Chord hold
    return (
        PadSynth().play(
            chord="i9",
            duration=16,  # Hold for 4 bars
        )
        >> Filter.LowPass(cutoff)
        >> Reverb(mix=0.4)
    )


# ==========================================
# Master Effects
# ==========================================
# Final chain before output


@session.master
def master_bus(audio_in):
    return audio_in >> Compressor(threshold=-10, ratio=4) >> Limiter()
```