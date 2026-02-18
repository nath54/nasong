from nasong.theory import render
from nasong.core.values.basic.value_identity import Identity
from nasong.instruments.synth import SynthLead
from nasong.theory.generators.styles.jazz import Jazz

# Configuration
BPM = 120
# Duration of a 4-bar loop (16 quarters)
LOOP_QUARTERS = 4
# (16 quarters / BPM) * 60 seconds
LOOP_SECONDS = (LOOP_QUARTERS * 60.0) / BPM

# 1. Define Looping Time: This is the secret for infinite loops!
time = Identity() % LOOP_SECONDS

# 3. Composition
prog = Jazz.generate_random_standards_progression(length=4)

# 4. Render to Sequencer
layer = render(prog, time, SynthLead, bpm=BPM)

# The final 'sequencer' variable is what the App looks for
sequencer = layer
