# Writing Songs with NaSong

This guide covers the standard (non-live) workflow for composing and rendering audio with NaSong.

---

## The Song Contract

Every NaSong script must define two module-level names:

```python
from nasong.core.value import Value

# Duration of the audio in seconds
duration: float = 10.0

# The song function
def song(time: Value) -> Value:
    """
    Args:
        time: A Value representing the current time in seconds.

    Returns:
        Value: The audio signal graph.
    """
    return ...
```

The CLI loads your script, reads `duration`, calls `song(time)` to build the audio graph, renders it sample-by-sample, and writes the result to a WAV file.

---

## Building Audio Graphs

### Oscillators

NaSong provides several waveform generators. Each takes a `time` value and a `frequency`:

```python
from nasong.core.all_values import Value, Sin, Sawtooth, Square, Triangle, Constant

# Duration of the audio in seconds
duration: float = 10.0

def song(time: Value) -> Value:
    # A 440 Hz sine wave
    # Sin expects frequency in radians/sec, so multiply Hz by 2π
    sine = Sin(value=time, frequency=Constant(440.0 * 6.2832))

    # Sawtooth, Square, Triangle use Hz directly
    saw  = Sawtooth(value=time, frequency=Constant(440.0))
    sq   = Square(value=time, frequency=Constant(440.0))
    tri  = Triangle(value=time, frequency=Constant(440.0))

    return sine  # pick one
```

> [!NOTE]
> `Sin` and `Cos` expect **radians per second** (`Hz × 2π`).
> `Sawtooth`, `Square`, and `Triangle` expect **Hz** directly.

### Combining Signals with `Sum`

Layer multiple signals by adding them:

```python
from nasong.core.all_values import Value, Sin, Constant, Sum

# Duration of the audio in seconds
duration: float = 10.0

def song(time: Value) -> Value:
    c4 = Sin(time, Constant(261.63 * 6.2832))
    e4 = Sin(time, Constant(329.63 * 6.2832))
    g4 = Sin(time, Constant(392.00 * 6.2832))

    return Sum([c4, e4, g4]) * Constant(0.3)
```

You can also use the `+` operator directly: `c4 + e4 + g4`.

### Controlling Volume with `Product`

Multiply a signal by a scalar or another value:

```python
from nasong.core.all_values import Value, Sin, Constant, Product

# Duration of the audio in seconds
duration: float = 10.0

def song(time: Value) -> Value:
    osc = Sin(time, Constant(440.0 * 6.2832))
    # Reduce volume to 30%
    return Product(osc, Constant(0.3))
    # Or equivalently:
    # return osc * Constant(0.3)
    # return osc * 0.3
```

---

## Amplitude Envelopes

Raw oscillators produce a constant volume. Use **envelopes** to shape dynamics.

### `ADSR2` — One-Shot Envelope

`ADSR2` is the workhorse envelope for individual notes:

```python
from nasong.core.all_values import ADSR2

env = ADSR2(
    time=time,
    note_start=0.0,         # When the note begins (seconds)
    note_duration=2.0,       # Total note duration (seconds)
    attack_time=0.05,        # Time to reach full volume
    decay_time=0.1,          # Time to fall to sustain level
    sustain_level=0.7,       # Volume during sustain (0.0–1.0)
    release_time=0.3,        # Fade-out time after note ends
)
```

### Applying an Envelope

Multiply the oscillator by the envelope:

```python
def song(time: Value) -> Value:
    osc = Sin(time, Constant(440.0 * 6.2832))
    env = ADSR2(time, 0.0, 3.0, 0.1, 0.2, 0.6, 0.5)
    return osc * env * 0.5
```

---

## Looping Patterns

Use the modulo operator on time to create loops:

```python
from nasong.core.all_values import Identity, Constant

# Loop every 2 seconds
loop_time = Identity() % Constant(2.0)
```

This wraps the global time so your pattern repeats. You can use `loop_time` wherever you would use `time`:

```python
def song(time: Value) -> Value:
    loop_dur = 2.0
    loop_time = time % Constant(loop_dur)
    osc = Sin(loop_time, Constant(440.0 * 6.2832))
    env = ADSR2(loop_time, 0.0, 1.0, 0.01, 0.1, 0.5, 0.3)
    return osc * env * 0.3
```

---

## The Sequencer — Playing Multiple Notes

The `Sequencer` takes a list of note data and an instrument factory, then places notes at the correct times:

```python
from nasong.core.all_values import Value, Sequencer, Sin, ADSR2, Constant

def my_instrument(time: Value, freq: float, start: float, duration: float, velocity: float) -> Value:
    osc = Sin(time, Constant(freq * 6.2832))
    env = ADSR2(time, start, duration, 0.01, 0.1, 0.5, 0.2)
    return osc * env * velocity * 0.5

duration: float = 5.0

def song(time: Value) -> Value:
    notes = [
        # (freq_hz, start_sec, duration_sec, velocity)
        (261.63, 0.0, 1.0, 0.8),   # C4
        (329.63, 1.0, 1.0, 0.7),   # E4
        (392.00, 2.0, 1.0, 0.9),   # G4
        (523.25, 3.0, 1.5, 1.0),   # C5
    ]
    return Sequencer(time, instrument_factory=my_instrument, note_data_list=notes)
```

### Using Theory's `render()` Instead

The `render()` function automates sequencer setup from theory objects:

```python
from nasong.core.value import Value
from nasong.theory import render
from nasong.theory.systems.western import Western
from nasong.theory.structures.progression import Progression
from nasong.theory.core.time import HALF

scale = Western.major("C4")
prog = Progression.from_roman_numerals(scale, ["I", "V", "vi", "IV"], duration=HALF)

seq = render(prog, time_value=None, instrument_factory=my_instrument, bpm=120)
```

See [Music Theory Module](music_theory.md) for details.

---

## Instrument Factory Signature

The instrument factory is called by the `Sequencer` for **each note**:

```python
def instrument_factory(time: Value, freq: float, start: float, duration: float, velocity: float) -> Value:
    """
    Args:
        time:     Global time Value (continuous, in seconds).
        freq:     Note frequency in Hz (float).
        start:    Note start time in seconds (float).
        duration: Note duration in seconds (float).
        velocity: Note velocity 0.0–1.0 (float).

    Returns:
        Value: The audio signal for this note.
    """
    ...
```

---

## Full Walkthrough — Multi-Instrument Song

Here's a complete example building a multi-instrument piece:

```python
from nasong.core.all_values import (
    Value, Sin, Sawtooth, ADSR2, WhiteNoise, Constant, Sum
)
from nasong.theory import render
from nasong.theory.systems.western import Western
from nasong.theory.structures.progression import Progression
from nasong.theory.core.time import QUARTER, HALF

duration: float = 8.0

# --- Instruments ---

def piano(time: Value, freq: float, start: float, dur: float, vel: float) -> Value:
    osc = Sin(time, Constant(freq * 6.2832))
    env = ADSR2(time, start, dur, 0.01, 0.15, 0.5, 0.3)
    return osc * env * vel * 0.4

def bass(time: Value, freq: float, start: float, dur: float, vel: float) -> Value:
    osc = Sawtooth(time, Constant(freq))
    env = ADSR2(time, start, dur, 0.01, 0.1, 0.7, 0.2)
    return osc * env * vel * 0.3

def hihat(time: Value, freq: float, start: float, dur: float, vel: float) -> Value:
    noise = WhiteNoise()
    env = ADSR2(time, start, 0.05, 0.001, 0.03, 0.0, 0.01)
    return noise * env * vel * 0.15

# --- Composition ---

def song(time: Value) -> Value:
    scale = Western.major("C4")

    # Piano: I - V - vi - IV
    piano_prog = Progression.from_roman_numerals(
        scale, ["I", "V", "vi", "IV"], duration=HALF
    )
    piano_seq = render(piano_prog, time, piano, bpm=120)

    # Bass: root notes
    bass_scale = Western.major("C3")
    bass_prog = Progression.from_roman_numerals(
        bass_scale, ["I", "V", "vi", "IV"], duration=HALF
    )
    bass_seq = render(bass_prog, time, bass, bpm=120)

    # Hi-hat: steady eighth notes
    hh_scale = Western.major("C5")
    hh_notes = ["I"] * 16
    hh_prog = Progression.from_roman_numerals(
        hh_scale, hh_notes, duration=QUARTER
    )
    hh_seq = render(hh_prog, time, hihat, bpm=120)

    return piano_seq + bass_seq + hh_seq
```

```bash
nasong multi_instrument.py -o song.wav
```

---

## Tips

- **Volume control**: Keep individual instrument levels around 0.3–0.5 to avoid clipping when summing.
- **Looping**: Use `time % Constant(loop_seconds)` to loop patterns.
- **BPM timing**: At 120 BPM, a quarter note = 0.5 seconds (`60 / BPM`).
- **PyTorch rendering**: Add `-t` for GPU-accelerated rendering when processing large files.
- **Explore examples**: Check the `nasong_examples/` directory for ready-to-run scripts.

---

## Related Documentation

- [Getting Started](getting_started.md) — Installation and first steps
- [Music Theory Module](music_theory.md) — Scales, chords, and progressions
- [Algo-Rave Guide](algo_rave.md) — Live coding with the TUI
- [Instruments](instruments.md) — Pre-built instrument library
- [Core Values API](api/core_values.md) — Complete Value node reference
