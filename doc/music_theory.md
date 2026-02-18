# Music Theory Module

The `nasong.theory` module provides a rich set of tools for working with musical structures — scales, chords, progressions, rhythms — and converting them into playable audio graphs.

---

## Core Concepts

### Pitch

Represents a musical frequency. Two concrete implementations:

| Class | Module | Description |
| :--- | :--- | :--- |
| `Hz(freq)` | `theory.core.pitch` | Raw frequency in Hertz |
| `Note(name)` | `theory.core.pitch` | Scientific pitch notation (e.g., `"C4"`, `"F#3"`, `"Bb5"`) |

```python
from nasong.theory.core.pitch import Hz, Note

a4 = Note("A4")
print(a4.freq)       # 440.0
print(a4.midi)       # 69

raw = Hz(261.63)
print(raw.to_hz())   # 261.63
```

**Key methods:**

| Method | Returns | Description |
| :--- | :--- | :--- |
| `to_hz()` | `float` | Frequency in Hz |
| `to_value()` | `Value` | A `Constant` wrapping the frequency |
| `transpose(semitones)` | `Note` | A new note shifted by *n* semitones |

### Interval

Represents the distance between two pitches:

```python
from nasong.theory.core.interval import Interval

fifth = Interval("P5")       # Perfect fifth = 7 semitones
print(fifth.semitones)       # 7.0
print(fifth.ratio)           # ~1.498

# Transpose a note
from nasong.theory.core.pitch import Note
c4 = Note("C4")
g4 = fifth.add_to(c4)       # Note("G4")
```

**Supported interval names:**

| Name | Semitones | Name | Semitones |
| :--- | :--- | :--- | :--- |
| `P1` / `unison` | 0 | `P5` / `perf5` | 7 |
| `m2` / `S` | 1 | `m6` | 8 |
| `M2` / `T` | 2 | `M6` | 9 |
| `m3` | 3 | `m7` | 10 |
| `M3` | 4 | `M7` | 11 |
| `P4` | 5 | `P8` / `octave` | 12 |
| `TT` / `tritone` | 6 | | |

### Duration

Symbolic note lengths expressed in **quarter notes**:

```python
from nasong.theory.core.time import (
    WHOLE, HALF, QUARTER, EIGHTH, SIXTEENTH, THIRTYSECOND,
    dotted, triplet
)

print(QUARTER.value)        # 1.0
print(EIGHTH.value)         # 0.5
print(dotted(QUARTER).value)  # 1.5
print(triplet(QUARTER).value) # ~0.667
```

| Constant | Value (in quarters) |
| :--- | :--- |
| `WHOLE` | 4.0 |
| `HALF` | 2.0 |
| `QUARTER` | 1.0 |
| `EIGHTH` | 0.5 |
| `SIXTEENTH` | 0.25 |
| `THIRTYSECOND` | 0.125 |

### TimeSignature

```python
from nasong.theory.core.time import TimeSignature

ts = TimeSignature(numerator=4, denominator=4)
print(ts.bar_length_in_quarters())  # 4.0

waltz = TimeSignature(6, 8)
print(waltz.bar_length_in_quarters())  # 3.0
```

---

## Structures

### NoteEvent (Note)

A single musical event with pitch, duration, and velocity:

```python
from nasong.theory.structures.note import Note as NoteEvent
from nasong.theory.core.pitch import Note as PitchNote
from nasong.theory.core.time import QUARTER

event = NoteEvent(
    pitch=PitchNote("C4"),
    duration=QUARTER,
    velocity=0.8
)
```

### Chord

A simultaneous group of notes:

```python
from nasong.theory.structures.chord import Chord

# From a list of NoteEvents
chord = Chord(notes=[event1, event2, event3])
# Or from a name
chord = Chord.from_name("Cmaj7")
```

### Progression

A sequence of chords:

```python
from nasong.theory.structures.progression import Progression
from nasong.theory.systems.western import Western
from nasong.theory.core.time import QUARTER

scale = Western.major("C4")

# Using Roman numerals
prog = Progression.from_roman_numerals(scale, ["I", "V", "vi", "IV"])

# With custom duration per chord
prog = Progression.from_roman_numerals(
    scale, ["I", "V", "vi", "IV"], duration=QUARTER
)
```

### Rhythm

A pattern of hits and rests:

```python
from nasong.theory.structures.rhythm import Rhythm

# From a string pattern (x = hit, . = rest)
clave = Rhythm.from_string("x..x..x...x.x...")
```

---

## Scale Systems

NaSong supports multiple musical traditions:

### Western

```python
from nasong.theory.systems.western import Western

scale = Western.major("C4")
scale = Western.minor("A4")
scale = Western.dorian("D4")
scale = Western.phrygian("E4")
scale = Western.lydian("F4")
scale = Western.mixolydian("G4")
scale = Western.locrian("B4")
scale = Western.mode("C4", "major")  # Generic mode access

# Dynamic note access
note = Western.C4   # Returns Note("C4")
note = Western.F3   # Returns Note("F3")
```

| Method | Scale Type |
| :--- | :--- |
| `major(root)` | Ionian (Major) |
| `minor(root)` | Aeolian (Natural Minor) |
| `dorian(root)` | Dorian mode |
| `phrygian(root)` | Phrygian mode |
| `lydian(root)` | Lydian mode |
| `mixolydian(root)` | Mixolydian mode |
| `locrian(root)` | Locrian mode |

### Raga (Indian Classical)

```python
from nasong.theory.systems.raga import Raga
```

### Maqam (Middle Eastern)

```python
from nasong.theory.systems.maqam import Maqam
```

### Gamelan (Indonesian)

```python
from nasong.theory.systems.gamelan import Gamelan
```

### African

```python
from nasong.theory.systems.african import African

scale = African.pentatonic("C4")
polyrhythm = African.polyrhythm((3, 2), length=12)
```

### East Asian

```python
from nasong.theory.systems.east_asian import EastAsian
```

---

## Style Generators

Pre-built generators for specific musical idioms. Each returns `Progression`, `Rhythm`, or both.

### Jazz

```python
from nasong.theory.generators.styles.jazz import Jazz

# Classic ii-V-I turnaround
prog = Jazz.ii_V_I(root="C4", minor=False)

# Random lead-sheet style progression
prog = Jazz.generate_random_standards_progression(length=8)
```

| Method | Parameters | Returns |
| :--- | :--- | :--- |
| `ii_V_I(root, minor)` | `root: str = "C4"`, `minor: bool = False` | `Progression` |
| `generate_random_standards_progression(length)` | `length: int = 4` | `Progression` |

### EDM

```python
from nasong.theory.generators.styles.edm import EDM

prog = EDM.epic_chords(root="F4")      # vi-IV-I-V
beat = EDM.basic_beat()                 # four-on-the-floor Rhythm
```

### Lofi

```python
from nasong.theory.generators.styles.lofi import Lofi

prog = Lofi.chill_progression(root="Db4")
```

### Salsa

```python
from nasong.theory.generators.styles.salsa import Salsa

prog = Salsa.montuno_progression(root="G4", minor=True)
clave = Salsa.clave_rhythm(direction="2-3")  # or "3-2"
```

### Afrobeat

```python
from nasong.theory.generators.styles.afrobeat import Afrobeat

prog = Afrobeat.polyrhythmic_groove(root="C4")
# Access polyrhythm lines:
# prog.rhythm_a  (3-beat line)
# prog.rhythm_b  (2-beat line)
```

### Celtic

```python
from nasong.theory.generators.styles.celtic import Celtic
```

### Bossa Nova

```python
from nasong.theory.generators.styles.bossa_nova import BossaNova
```

### Koto

```python
from nasong.theory.generators.styles.koto import Koto
```

---

## The `render()` Function

The bridge between theory objects and audio. Converts a `Progression`, `Chord`, or `NoteEvent` into a `Sequencer` Value node:

```python
from nasong.theory import render

sequencer = render(
    obj,                    # Progression, Chord, or NoteEvent
    time_value,             # A time Value (or None for auto Identity)
    instrument_factory,     # function(time, freq, start, dur, vel) -> Value
    bpm=120.0               # Beats per minute
)
```

**How it works:**

1. Converts the theory object into a list of `(freq, start_sec, dur_sec, velocity)` tuples
2. Calculates timing from BPM (quarter note = `60 / BPM` seconds)
3. Creates a `Sequencer` that calls `instrument_factory` for each note

### The Instrument Factory

```python
def my_synth(time, freq, start, duration, velocity):
    """
    Args:
        time:     Global time Value (continuous seconds).
        freq:     Note frequency in Hz.
        start:    Note start time in seconds.
        duration: Note duration in seconds.
        velocity: Note velocity (0.0–1.0).
    Returns:
        Value: Audio signal for this note.
    """
    osc = Sawtooth(value=time, frequency=Constant(freq))
    env = ADSR2(time, start, duration, 0.01, 0.1, 0.7, 0.2)
    return osc * env * Constant(velocity * 0.5)
```

---

## Examples

### ii-V-I Jazz Progression

```python
from nasong.theory.generators.styles.jazz import Jazz
from nasong.theory import render
from nasong.core.all_values import Sin, ADSR2, Constant

def beep(time, freq, start, dur, vel):
    env = ADSR2(time, start, dur, 0.01, 0.1, 0.5, 0.1)
    return Sin(time, Constant(freq * 6.2832)) * env * vel * 0.5

prog = Jazz.ii_V_I("C4")
sequencer = render(prog, time_value=None, instrument_factory=beep, bpm=100)
```

### EDM Drop

```python
from nasong.theory.generators.styles.edm import EDM
from nasong.theory import render

prog = EDM.epic_chords("F4")
sequencer = render(prog, time_value=None, instrument_factory=my_synth, bpm=128)
```

### Lofi Beat

```python
from nasong.theory.generators.styles.lofi import Lofi
from nasong.theory import render

prog = Lofi.chill_progression("Db4")
sequencer = render(prog, time_value=None, instrument_factory=my_synth, bpm=75)
```

---

## Related Documentation

- [Song Scripting Guide](song_scripting.md) — Building songs step by step
- [Core Values API](api/core_values.md) — Available Value nodes
- [Style Generators API](api/theory_generators.md) — Detailed per-style reference
- [Instruments](instruments.md) — Pre-built instrument library
