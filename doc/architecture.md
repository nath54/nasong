# Project Architecture

This document provides a high-level overview of NaSong's codebase for developers and contributors.

---

## Module Diagram

```
nasong/
├── core/                  ← The engine
│   ├── value.py           ← Value base class & ParameterContext
│   ├── all_values.py      ← Consolidated value registry
│   ├── config.py          ← Config (sample rate, duration, output)
│   ├── song.py            ← Song rendering (NumPy & PyTorch)
│   ├── wav.py             ← WAV file I/O utilities
│   ├── utils.py           ← Module import helpers
│   ├── vis.py             ← Audio visualization tools
│   └── values/            ← All Value node implementations
│       ├── basic/         ← Constant, Identity, WhiteNoise, Random*
│       ├── complex/       ← Sin, Cos, ADSR2, Distortion, ...
│       ├── single_itms_ops/  ← Abs, Clamp, Filter, Sequencer, ...
│       ├── mult_itms_ops/    ← Sum, Product, Max, Min, ...
│       ├── formant.py     ← Vocal formant synthesis
│       ├── utils.py       ← generate_harmonics, LFO
│       └── music_theory_and_composition.py  ← SimpleMelody, midi helpers
│
├── instruments/           ← Pre-built instrument factories
│   ├── bass.py            ← SynthBass, AcousticBass, ...
│   ├── keyboards.py       ← GrandPiano, ElectricPiano, ...
│   ├── synth.py           ← SynthLead, SynthPad, ...
│   ├── percussion.py      ← KickDrum, SnareDrum, HiHat, ...
│   ├── bowed_strings.py   ← Violin, Cello, ...
│   ├── plucked_strings.py ← AcousticGuitar, ElectricGuitar, ...
│   ├── winds.py           ← Flute, Trumpet, ...
│   └── effects.py         ← ADSR_Piano, Vibrato
│
├── theory/                ← Music theory layer
│   ├── __init__.py        ← render(), expand(), track_from_progression()
│   ├── core/              ← Pitch, Interval, Scale, Duration, TimeSignature
│   ├── structures/        ← Note, Chord, Progression, Rhythm
│   ├── systems/           ← Western, Raga, Maqam, Gamelan, African, EastAsian
│   └── generators/
│       └── styles/        ← Jazz, EDM, Lofi, Salsa, Afrobeat, Celtic, ...
│
├── dsl/                   ← Optional fluent DSL
│   ├── chain.py           ← Chainable, Processor, Gain
│   ├── decorators.py      ← @instrument, @effect
│   └── units.py           ← BPM, Ms, Bars, Hz
│
├── app/                   ← Live coding application
│   ├── live_session.py    ← LiveSession: hot-reload + audio stream
│   ├── render_engine.py   ← RenderEngine: background chunk rendering
│   ├── main_tui.py        ← AlgoRaveApp: Textual TUI
│   ├── main_headless.py   ← Headless (no UI) runner
│   ├── main_daw.py        ← DAW integration mode
│   └── docs_utils.py      ← API introspection for the doc browser
│
├── trainable/             ← Trainable instrument framework
│   └── instruments/       ← Neural-parameterized instruments
│
├── scripts/               ← CLI entry points
│   ├── main.py            ← `nasong` CLI
│   ├── train_cli.py       ← `nasong-train` CLI
│   ├── evaluate.py        ← Model evaluation pipeline
│   ├── experiment_manager.py ← Experiment tracking
│   ├── leaderboard.py     ← Training leaderboard
│   └── vis_tool.py        ← Visualization CLI
│
└── dummy/                 ← Stub/test modules
```

---

## Data Flow: Script → Sound

### Standard CLI (`nasong script.py`)

```
Python Script
    │
    │  import + exec
    ▼
┌─────────────────────┐
│ module.song(time)    │ ← User defines this function
│ module.duration      │ ← User sets audio length
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Song(config, fn)     │ ← Wraps config + song function
│   .render()          │ ← Creates Identity → BasicScaling → time
│   .getitem_np(buf)   │ ← Evaluates the full Value graph
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ WavUtils             │ ← prepare_signal() + save_wav_file()
└─────────────────────┘
```

### Live Mode (`nasong-rave script.py`)

```
Python Script
    │
    │  load_script()
    ▼
┌─────────────────────┐
│ LiveSession           │  ← Manages audio stream & hot-reload
│   .load_script()      │  ← Extracts `sequencer` or `song(time)`
│   .audio_callback()   │  ← Feeds samples to PortAudio
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│ RenderEngine          │  ← Background thread
│   .set_sequencer()    │  ← Receives Value graph
│   ._render_loop()     │  ← Continuously renders chunks
│   .get_audio_chunk()  │  ← Returns cached chunk to callback
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│ sounddevice stream    │  ← PortAudio → speakers 🔊
└─────────────────────┘
```

---

## Key Components

### `RenderEngine`

The render engine runs in a **background thread** and pre-computes audio chunks using a **priority queue**:

- Chunks near the playback cursor get highest priority
- Rendered chunks are stored in a **cache** keyed by `(start_sample, version_id)`
- **Double-buffering**: When `FORCE_RERENDER_EVERY` triggers a re-render, the old cache entries continue serving audio while fresh ones are computed
- **Version tracking**: Each cache invalidation increments a version counter; stale chunks are eventually replaced

### `LiveSession`

The bridge between the audio stream, the user script, and the TUI:

- **Hot-reload**: Re-imports the Python script on demand, capturing the new `sequencer` variable
- **Audio callback**: Called by PortAudio; pulls chunks from `RenderEngine`, applies master volume, writes to output buffer
- **Error handling**: Script errors are caught and reported via callbacks without crashing the audio stream
- **Logging**: Captures `print()` output from user scripts via a custom stream redirector

### TUI vs. Headless vs. DAW

| Mode | Module | UI | Use Case |
| :--- | :--- | :--- | :--- |
| TUI | `main_tui.py` | Full Textual app | Interactive live coding |
| Headless | `main_headless.py` | None | Performance, CI |
| DAW | `main_daw.py` | DAW integration | Plugin workflow |

All three modes use the same `LiveSession` and `RenderEngine` under the hood.

### Training Pipeline

```
Audio Reference (.wav)
    │
    │  Note Detection (CREPE / basic_pitch / etc.)
    ▼
┌─────────────────────┐
│ Note Event List       │  ← (freq, start, dur, vel) tuples
└──────────┬───────────┘
           │
           │  Build Value graph with ValueTrainableParameter nodes
           ▼
┌─────────────────────┐
│ Value Graph           │  ← Instrument + envelope + effects
│   with trainable      │
│   parameters          │
└──────────┬───────────┘
           │
           │  Render (PyTorch) → Spectral Loss (STFT)
           ▼
┌─────────────────────┐
│ Optimizer             │  ← Adam / gradient descent
│   .backward()         │  ← Gradients flow through Value graph
│   .step()             │  ← Update ValueTrainableParameter values
└──────────┬───────────┘
           │
           │  Repeat for N epochs
           ▼
┌─────────────────────┐
│ Trained Instrument    │  ← Optimized parameter values
└─────────────────────┘
```

### Experiment Tracking

| Module | Purpose |
| :--- | :--- |
| `experiment_manager.py` | Run management, checkpointing, metric logging |
| `leaderboard.py` | Cross-experiment comparison and ranking |
| `evaluate.py` | Post-training evaluation pipeline |
| `monitor_cli.py` | Real-time training monitoring |

---

## Related Documentation

- [Value System](value_system.md) — Deep dive into the Value architecture
- [Contributing](contributing.md) — How to extend the codebase
- [Training Config](training_config.md) — YAML config reference for training
