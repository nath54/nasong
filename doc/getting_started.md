# Getting Started

Welcome to **NaSong** — a Python-based, code-first audio synthesizer and music composition engine.
This guide will get you from zero to audible output in under five minutes.

---

## Prerequisites

| Requirement | Version |
| :--- | :--- |
| Python | ≥ 3.10 |
| pip | any recent |

Optional (for hardware-accelerated rendering):

- **PyTorch** — enables `--torch` rendering and gradient-based training.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/nath54/nasong.git
cd nasong
```

### 2. Install in editable mode

```bash
pip install -e .
```

This makes the `nasong` and `nasong-rave` CLI commands available globally.

> [!TIP]
> Create a virtual environment first to keep your global packages clean:
> ```bash
> python -m venv .venv
> source .venv/bin/activate   # Linux / macOS
> .venv\Scripts\activate      # Windows
> pip install -e .
> ```

---

## Hello World — A 440 Hz Sine Wave

Create a file called `hello.py`:

```python
from nasong.core.all_values import Sin, Constant, Identity

# Duration of our audio clip in seconds
duration = 3.0

# The song function: takes a time Value and returns an audio Value
def song(time):
    return Sin(value=time, frequency=Constant(440.0 * 6.2832))
```

Render it to a WAV file:

```bash
nasong hello.py -o hello.wav
```

You now have a 3-second, 440 Hz sine wave saved as `hello.wav`. 🎉

---

## Understanding the Two Key Concepts

Every NaSong script must expose two things:

| Variable | Type | Purpose |
| :--- | :--- | :--- |
| `duration` | `float` | Length of the audio in seconds |
| `song` | `function(time) → Value` | A function that receives a time `Value` and returns the audio signal tree |

The `time` argument is a continuously increasing value (in seconds). You build an audio graph from it using `Value` nodes like `Sin`, `Sum`, `Product`, `ADSR2`, etc.

---

## A More Interesting Example

Let's create a chord with an amplitude envelope:

```python
from nasong.core.all_values import (
    Sin, Constant, Sum, Product, Identity, ADSR2, BasicScaling
)

duration = 5.0

def song(time):
    # Build three sine waves for a C major chord (C4, E4, G4)
    c4 = Sin(value=time, frequency=Constant(261.63 * 6.2832))
    e4 = Sin(value=time, frequency=Constant(329.63 * 6.2832))
    g4 = Sin(value=time, frequency=Constant(392.00 * 6.2832))

    # Mix and attenuate
    chord = Sum([c4, e4, g4]) * Constant(0.3)

    # Apply a simple envelope
    env = ADSR2(
        time=time,
        note_start=0.0,
        note_duration=duration,
        attack_time=0.5,
        decay_time=0.5,
        sustain_level=0.6,
        release_time=1.0,
    )

    return chord * env
```

```bash
nasong chord.py -o chord.wav
```

---

## CLI Reference

```
nasong <script.py> [OPTIONS]

Positional:
  script.py              Path to a NaSong song script

Options:
  -o, --output FILE      Output WAV path (default: output.wav)
  -s, --sample-rate INT  Sample rate in Hz (default: 44100)
  -t, --torch            Use PyTorch backend for rendering
  -d, --device DEVICE    Torch device (default: cpu)
```

---

## Next Steps

| Topic | Document |
| :--- | :--- |
| Building full songs step by step | [Song Scripting Guide](song_scripting.md) |
| Using scales, chords, and progressions | [Music Theory Module](music_theory.md) |
| Live coding with the TUI | [Algo-Rave Guide](algo_rave.md) |
| All available Value nodes | [Core Values API](api/core_values.md) |
| Instrument library reference | [Instruments](instruments.md) |
| Training custom instruments | [Training Config](training_config.md) |
