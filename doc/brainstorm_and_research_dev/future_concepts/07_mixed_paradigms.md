# 07_mixed_paradigms.py

```py
# DEMONSTRATION: The "Polyglot" Engine
# This file proves that we can support MULTIPLE user styles in one session.
# The `Session` object acts as a generic host for different "Sequencer Strategies".

from nasong.live import Session

# Import specific syntax adapters
from nasong.syntax import imperative, patterns, functional
from nasong.instruments import Kick, Bass, Pad

session = Session(bpm=128)


# ==========================================
# Style 1: The "Sonic Pi" Imperative Style
# Best for: Linear rhythms, storytelling, intuitive flow.
# ==========================================
@session.track(style=imperative)
def drums(ctx):
    # This runs in a verified coroutine/generator
    while True:
        play(Kick)
        wait(0.5)  # Wait half beat
        play(Kick, velocity=0.8)
        wait(0.25)
        play(Kick, velocity=0.5)
        wait(0.25)


# ==========================================
# Style 2: The "Tidal" Pattern Style
# Best for: Complex polyrhythms, algorithmic variation, brevity.
# ==========================================
@session.track(style=patterns)
def bassline():
    # We return a Dictionary of Pattern Objects
    return {
        "voice": Bass(),
        # Euclidean Rhythm: 3 hits in 8 steps
        # "< >" means "cycle per bar"
        "notes": P("<C2 F2 G2 C2>"),
        "rhythm": P("x(3,8)"),
        "filter": P("400 800").smooth(),  # Automation
    }


# ==========================================
# Style 3: The "NaSong" Functional Style (Standard)
# Best for: Drone, Texture, pure Physics/Math modulation.
# ==========================================
@session.track(style=functional)
def texture(t):
    # 't' is the raw Time Vector for the current chunk
    # This gives you raw, sample-level control

    # Pure math synthesis
    mod = Sine(t * 0.1) * 500
    signal = Pad(freq=440 + mod)

    return signal


# ==========================================
# How it works internally
# ==========================================
# The Session loops over all tracks.
# - If style is 'imperative': it steps the generator until it hits a 'wait',
#   schedules those events into the Audio Engine's event queue.
# - If style is 'patterns': it queries the pattern string for the current
#   time window (t_start, t_end) and schedules resulting events.
# - If style is 'functional': it just calls the function(t) and mixes the
#   raw audio output directly.
#
# Result: They all output audio/events to the same bus!
```