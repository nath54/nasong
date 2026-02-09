# 01_theory_basics.py
# This file demonstrates the core Music Theory API for NaSong.

from nasong.theory.core import Note, Scale, Interval, Hz
from nasong.theory.structures import Chord, Progression
from nasong.theory.systems import Western

# ==========================================
# 1. Pitch & Frequencies
# ==========================================

# Standard note creation
a4 = Note("A4")  # defaults to 440Hz
print(a4.freq)  # <Hz: 440.0>

# Microtonal/Custom Tuning Support
# We can define a note by explicit frequency or cents
micro_tone = Note(Hz(445))  # Slightly sharp A4

# Converting between representations
c5 = Note("C5")
print(c5.midi)  # 72
print(Western.freq(72))  # 523.25 Hz

# ==========================================
# 2. Scales (The foundation of tonality)
# ==========================================

# Create a C Major scale
c_major = Scale("C", "major")

# Querying the scale
print(c_major.notes)  # [C, D, E, F, G, A, B]
print(c_major.intervals)  # [0, 2, 4, 5, 7, 9, 11] (semitones)

# Getting specific degrees (1-based index)
print(c_major.degree(1))  # Note("C")
print(c_major.degree(5))  # Note("G")

# Advanced: Exotic Scales
# Defined by interval steps [2, 1, 2, ...]
hirajoshi = Scale("A", [2, 1, 4, 1, 4])
print(hirajoshi.notes)  # Pentatonic scale notes

# ==========================================
# 3. Chords (Vertical Harmony)
# ==========================================

# Factory methods for common chords
c_maj7 = Chord("Cmaj7")
d_min9 = Chord("Dm9")

# Voicings
# Default is closed position (root position)
print(c_maj7.notes)  # [C4, E4, G4, B4]

# Inversions
print(c_maj7.invert(1))  # [E4, G4, B4, C5]

# Open Voicings (Drop 2, Drop 3) for better orchestration
# "Drop 2" moves the 2nd highest note down an octave
jazz_voicing = c_maj7.voice("drop2")

# Polychords / Slash Chords
complex_harmony = Chord("D/C")  # D Major triad over C bass

# ==========================================
# 4. Progressions (Horizontal Harmony)
# ==========================================

# Define a chord progression using Roman Numerals
# This is context-dependent (needs a Key)
pop_prog = Progression("I V vi IV")

# Resolve it to a specific key
resolved = pop_prog.resolve(key="G", scale="major")
# Result: [G Major, D Major, E Minor, C Major]

# Jazz "Rhythm Changes" A-Section
rhythm_changes = Progression("I vi ii V | iii vi ii V | I I7 IV #ivdim7 | I V I")
print(rhythm_changes.resolve("Bb"))

# ==========================================
# 5. Integration with NaSong Values
# ==========================================
# All theory objects can be converted to something NaSong understands

# Convert a chord to a list of frequency Values
oscillator_freqs = c_maj7.to_values()
# [Value(261.6), Value(329.6), Value(392.0), Value(493.8)]

# This allows:
# song = Mix([Osc.Sine(f) for f in oscillator_freqs])
