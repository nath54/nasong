# Proposed Documentation Plan

This document proposes a comprehensive documentation structure for the NaSong project, organized by audience.

---

## Current State

| Document | Status |
| :--- | :--- |
| `README.md` | ✅ Exists — high-level overview, installation, CLI commands |
| `doc/training_config.md` | ✅ Exists — YAML config reference |
| `doc/instruments.md` | ✅ Exists — normal & trainable instruments |

---

## Proposed Structure

```
doc/
├── getting_started.md          # For newcomers
├── instruments.md              # ✅ Done
├── training_config.md          # ✅ Done
├── value_system.md             # Core architecture
├── music_theory.md             # Theory module reference
├── algo_rave.md                # Live coding guide
├── song_scripting.md           # Standard nasong scripting
├── dsl_reference.md            # DSL syntax reference
├── architecture.md             # For developers
├── contributing.md             # For contributors
└── api/                        # Auto-generated or manual API docs
    ├── core_values.md
    ├── effects.md
    └── theory_generators.md
```

---

## 1. For Curious / New Visitors

### `getting_started.md` — Quick Start Guide

**Audience**: Someone who just discovered the project and wants to hear sound in 5 minutes.

**Contents**:
- Installation (copy from README, keep minimal)
- "Hello World" — a 3-line script that plays a sine wave
- Running `nasong my_song.py -o hello.wav`
- Next steps: links to other docs

### `song_scripting.md` — Writing Songs with NaSong

**Audience**: Users who want to compose music programmatically.

**Contents**:
- The `song(time)` / `duration` contract
- Building audio graphs step by step
- Layering instruments with `Sum`
- Using `Sequencer` manually vs. `render()`
- Controlling volume with `Product`
- Looping patterns with `Identity() % loop_duration`
- Full walkthrough: building a multi-instrument song from scratch

---

## 2. For Active Users

### `instruments.md` — Instrument Reference ✅

Already written. Covers normal instruments, trainable instruments, Sequencer contract, and usage examples.

### `training_config.md` — Training Configuration ✅

Already written. YAML config reference for the training pipeline.

### `music_theory.md` — Music Theory Module

**Audience**: Users who want to use scales, chords, progressions, and style generators.

**Contents**:
- Core concepts: `Pitch`, `Interval`, `Duration`, `NoteEvent`
- Structures: `Chord`, `Progression`
- Systems: `Western` (major/minor/modes), `Raga`, `Maqam`, `Gamelan`
- *(table of available scales per system)*
- Style generators: `Jazz`, `EDM`, `Lofi`, `Salsa`, `Afrobeat`
- *(table of generator functions with signatures)*
- `render()` function — from theory objects to audio
- Examples: ii-V-I jazz, EDM drop, lofi beat

### `algo_rave.md` — Live Coding with Algo-Rave

**Audience**: Users who want to use the `nasong-rave` TUI for live coding sessions.

**Contents**:
- Launching: `nasong-rave` and `nasong-rave --headless`
- The live script contract: `sequencer` variable, `Identity() % LOOP_SECONDS`
- Hot-reloading: how it works, what triggers reload
- Keyboard shortcuts reference (table)
- `FORCE_RERENDER_EVERY` — double-buffering for randomness
- Mixing multiple layers: `sequencer = layer1 + layer2 + ...`
- Tips: looping, BPM changes, volume control, instrument switching
- Walkthrough of included examples (`01_techno_kick.py`, etc.)

### `dsl_reference.md` — DSL Syntax Reference

**Audience**: Users who want to use the optional NaSong DSL (if applicable).

**Contents**:
- DSL syntax overview
- Comparison with Python API
- Examples
- Limitations

---

## 3. For Developers / Contributors

### `value_system.md` — The Value Architecture

**Audience**: Developers who want to understand or extend the core engine.

**Contents**:
- Philosophy: "Everything is a Value"
- The `Value` base class: `get_item()`, `getitem_np()`, `getitem_torch()`
- Value categories:
  - **Basic**: `Constant`, `Identity`, `WhiteNoise`
  - **Single-Item Ops**: `Sin`, `Product`, `BasicScaling`, `ADSR2`, `ExponentialADSR`, `Sequencer`
  - **Multi-Item Ops**: `Sum`, `Concatenation`
  - **Training**: `ValueTrainableParameter`
- How to create a new Value node (step-by-step)
- The rendering pipeline: `Song` → chunk-based rendering → WAV
- NumPy vectorization: how `getitem_np` works
- PyTorch path: `getitem_torch` and autograd
- The `backward()` method and custom gradient support

### `architecture.md` — Project Architecture

**Audience**: New developers joining the project.

**Contents**:
- High-level module diagram (core, instruments, theory, trainable, app, scripts)
- Data flow: script → module loading → Value graph → chunk rendering → audio callback
- `RenderEngine`: caching, background rendering, priority queue
- `LiveSession`: script hot-reloading, audio callback, stream management
- TUI vs. Headless vs. DAW: the three app modes
- Training pipeline: note detection → graph construction → spectral loss → optimization
- Experiment tracking system

### `contributing.md` — Contributor Guide

**Audience**: People who want to contribute code.

**Contents**:
- Development setup (editable install, venv, requirements)
- Code style: Google-style docstrings, type hints, Pylint/Mypy
- Running tests: `pytest`, test generation script
- Adding a new instrument (step-by-step checklist)
- Adding a new Value node
- Adding a new music theory system
- Adding a new style generator
- PR guidelines

### `api/core_values.md` — Value Nodes Reference

**Audience**: Developers and advanced users.

**Contents**:
- Table of all available Value nodes with signatures
- Grouped by category (oscillators, envelopes, math, noise, sequencing)
- Parameter descriptions and defaults
- Usage examples for each

### `api/effects.md` — Effects Reference

**Audience**: Users and developers.

**Contents**:
- `ADSR_Piano` (looping envelope)
- `Vibrato` (FM generator)
- `Distortion`, `BandLimitedSawtooth`, `BandLimitedSquare`, `LFO`
- Usage examples

### `api/theory_generators.md` — Style Generators API

**Audience**: Users who want to use or extend style generators.

**Contents**:
- Per-style documentation: Jazz, EDM, Lofi, Salsa, Afrobeat
- Available methods per style class
- Parameter tables
- Output types and how to feed them to `render()`

---

## Priority Order

Recommended implementation order based on user impact:

| Priority | Document | Impact |
| :--- | :--- | :--- |
| 🔴 High | `getting_started.md` | First thing new users need |
| 🔴 High | `algo_rave.md` | Active feature, undocumented |
| 🟡 Medium | `music_theory.md` | Rich module, many generators |
| 🟡 Medium | `value_system.md` | Essential for developers |
| 🟡 Medium | `song_scripting.md` | Core use case guide |
| 🟢 Low | `architecture.md` | For onboarding new devs |
| 🟢 Low | `contributing.md` | For open-source contributors |
| 🟢 Low | `dsl_reference.md` | Only if DSL is actively used |
| 🟢 Low | `api/*` | Can be auto-generated later |
