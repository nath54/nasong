# Live Coding with Algo-Rave

NaSong includes a full-featured live-coding environment called **Algo-Rave**, built with [Textual](https://github.com/Textualize/textual). It provides a TUI (terminal user interface) for real-time audio programming with hot-reloading, volume control, and an integrated documentation browser.

---

## Launching

### TUI Mode (default)

```bash
nasong-rave [script.py]
```

Opens the full TUI with a code editor, transport controls, documentation browser, and log viewer.

### Headless Mode

```bash
nasong-rave --headless script.py
```

Runs without the TUI — just starts the audio stream and plays the script. Useful for performances or when running from a minimal terminal.

### CLI Options

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--headless` | Run without the TUI | Off |
| `--device DEVICE` | Audio output device | System default |
| `--sample-rate SR` | Sample rate in Hz | 44100 |
| `--block-size BS` | Audio buffer size | 2048 |
| `--volume VOL` | Initial volume (0.0–1.0) | 0.8 |

---

## The Live Script Contract

A live-rave script must define a single top-level variable:

```python
sequencer = <Value>
```

The `sequencer` variable should be a NaSong `Value` node representing the audio output. Alternatively, you can define a `song(time)` function (same as the standard CLI), but `sequencer` is preferred for live use.

### Minimal Example

```python
from nasong.core.all_values import Sin, Constant, Identity

# A simple 440 Hz tone
sequencer = Sin(value=Identity(), frequency=Constant(440.0 * 6.2832)) * 0.3
```

### Using Theory and Instruments

```python
from nasong.core.all_values import Sin, ADSR2, Constant
from nasong.theory import render
from nasong.theory.systems.western import Western
from nasong.theory.structures.progression import Progression
from nasong.theory.core.time import QUARTER

def beep(time, freq, start, dur, vel):
    env = ADSR2(time, start, dur, 0.01, 0.1, 0.5, 0.1)
    return Sin(time, Constant(freq * 6.2832)) * env * vel * 0.5

scale = Western.major("C2")
prog = Progression.from_roman_numerals(scale, ["I", "I", "I", "I"], duration=QUARTER)
long_prog = Progression(prog.chords * 16, [QUARTER] * (4 * 16))

sequencer = render(long_prog, time_value=None, instrument_factory=beep, bpm=130)
```

---

## Hot-Reloading

The TUI watches your script file for changes. When you:

1. **Save the file** (`Ctrl+S` in the editor, or save from your external editor)
2. **Press Reload** (`Ctrl+R`)

The engine will:
- Re-execute your script
- Extract the new `sequencer` variable
- Seamlessly swap the audio source

> [!TIP]
> The audio keeps playing during reload. If your new code has an error, the old audio continues and the error is displayed in the log panel.

---

## `FORCE_RERENDER_EVERY` — Double-Buffering for Randomness

If your script uses randomization (e.g., `random.choice`, `RandomFloat`, etc.), the randomness is frozen when the script is loaded. To get evolving patterns, define:

```python
FORCE_RERENDER_EVERY = 8  # Number of chunks before re-executing the script
```

This causes the engine to periodically re-run your script, generating fresh random values. The old audio continues playing while the new version is rendered in the background (double-buffering), so there are no audible gaps.

```python
import random
from nasong.core.all_values import Sin, Constant, Identity

FORCE_RERENDER_EVERY = 4

freq = random.choice([220, 330, 440, 550]) * 6.2832
sequencer = Sin(Identity(), Constant(freq)) * 0.3
```

---

## Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Ctrl+S` | Save current file |
| `Ctrl+R` | Reload / re-execute script |
| `Ctrl+L` | Toggle log screen |
| `Ctrl+Q` | Quit the application |

### Transport Controls (in the TUI)

The TUI provides on-screen buttons for:
- **Play / Stop** — Start or stop the audio stream
- **Reload** — Re-execute the script
- **Volume slider** — Adjust master volume
- **BPM slider** — Adjust tempo (visible when applicable)

---

## Mixing Multiple Layers

Sum multiple sequencers to create a layered mix:

```python
from nasong.core.all_values import Sum

# Each layer is a rendered sequencer
seq_kick = render(kick_prog, None, kick_drum, bpm=130)
seq_bass = render(bass_prog, None, bass_synth, bpm=130)
seq_hats = render(hat_prog, None, hihat, bpm=130)

sequencer = seq_kick + seq_bass + seq_hats
```

---

## Tips & Best Practices

### Looping

Use time modulo to create seamless loops:

```python
from nasong.core.all_values import Identity, Constant

LOOP_SECONDS = 4.0  # 4 bars at 120 BPM

# Wrap time so your pattern repeats
time = Identity() % Constant(LOOP_SECONDS)
```

### Volume Control

Keep individual layer volumes low (0.2–0.5) to avoid clipping when summing:

```python
sequencer = (kick * 0.6) + (bass * 0.4) + (hats * 0.2)
```

### BPM Changes

Calculate beat duration from BPM:

```python
BPM = 128
BEAT_SEC = 60.0 / BPM        # 0.46875s per beat
BAR_SEC = BEAT_SEC * 4         # 1.875s per bar
```

### Instrument Switching

Define multiple instruments and swap them by editing your script and pressing `Ctrl+R`:

```python
# Try different oscillators by changing this line:
osc = Sawtooth(time, Constant(freq))   # Buzzy
# osc = Sin(time, Constant(freq * 6.2832))  # Clean
# osc = Square(time, Constant(freq))   # Hollow
```

---

## Included Examples Walkthrough

The `nasong_examples/live_rave/` directory contains ready-to-play scripts:

| Script | Description |
| :--- | :--- |
| `01_techno_kick.py` | Basic 4/4 techno kick pattern with pitch sweep |
| `02_ambient_drone.py` | Layered sine drones with LFO modulation |
| `03_generative_melody.py` | Randomly generated melodic sequences |
| `04_a.py` | Experimental pattern |

Launch any of them:

```bash
nasong-rave nasong_examples/live_rave/01_techno_kick.py
```

---

## How It Works Under the Hood

```
┌──────────────────────────┐
│     Your Script (.py)     │
│   sequencer = <Value>     │
└─────────┬────────────────┘
          │ load_script()
          ▼
┌──────────────────────────┐
│      LiveSession          │
│  - Hot-reload manager     │
│  - Audio stream control   │
│  - Volume / seek          │
└─────────┬────────────────┘
          │ set_sequencer()
          ▼
┌──────────────────────────┐
│      RenderEngine         │
│  - Background thread      │
│  - Chunk-based rendering  │
│  - Priority cache         │
│  - Double-buffering       │
└─────────┬────────────────┘
          │ audio_callback()
          ▼
┌──────────────────────────┐
│  PortAudio (sounddevice)  │
│  → Your speakers 🔊      │
└──────────────────────────┘
```

---

## Related Documentation

- [Getting Started](getting_started.md) — Installation and first render
- [Song Scripting Guide](song_scripting.md) — Standard (non-live) composition
- [Music Theory Module](music_theory.md) — Scales, chords, and progressions
- [Instruments](instruments.md) — Pre-built instrument library
