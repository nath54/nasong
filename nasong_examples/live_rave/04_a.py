from nasong.theory import render
from nasong.core.values.basic.value_identity import Identity
from nasong.instruments.synth import SynthLead
from nasong.instruments.keyboards import PianoNote
from nasong.theory.generators.styles.jazz import Jazz

# Configuration
BPM = 120
# Duration of a 4-bar loop (16 quarters)
LOOP_QUARTERS = 4
# (16 quarters / BPM) * 60 seconds
LOOP_SECONDS = (LOOP_QUARTERS * 60.0) / BPM

# Force re-render every N chunks to get fresh random progressions
# FORCE_RERENDER_EVERY = 256

# 1. Define Looping Time: This is the secret for infinite loops!
time = Identity() % LOOP_SECONDS

# 3. Composition
prog = Jazz.generate_random_standards_progression(length=4)
prog2 = Jazz.generate_random_standards_progression(length=4)

# 4. Render to Sequencer
layer = render(prog, time, SynthLead, bpm=BPM)
layer2 = render(prog2, time, PianoNote, bpm=BPM)

# The final 'sequencer' variable is what the App looks for
sequencer = layer + layer2
