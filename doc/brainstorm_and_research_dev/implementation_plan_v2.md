# NaSong Structure & Implementation Plan v2

This document outlines the architectural plan for integrating comprehensive music theory systems and a pseudo-realtime "algo-rave" engine into NaSong, replacing the initial research document with a concrete implementation strategy.

## 1. Music Theory System Architecture

We will implement a layered architecture that separates mathematical precision, musical abstraction, and genre-specific stylization.

### 1.1 Code Structure Organization

The `nasong.theory` module will be organized as follows:

```
src/nasong/
└── theory/
    ├── __init__.py
    ├── core/                 # Fundamental mathematical & musical units
    │   ├── pitch.py          # Frequencies, MIDI, Tuning systems
    │   ├── interval.py       # Intervallic math, ratios, cents
    │   ├── time.py           # TimeSignatures, ticks, rhythmic math
    │   └── scale.py          # Generic Scale class (steps, intervals)
    ├── structures/           # Musical building blocks
    │   ├── note.py           # Note events (pitch + duration + velocity)
    │   ├── chord.py          # Chord construction, voicings
    │   ├── progression.py    # Roman numeral analysis, functional harmony
    │   └── rhythm.py         # Patterns, euclidean rhythms, swing
    ├── systems/              # Cultural & Rule-based systems
    │   ├── western.py        # Major/Minor, Modes
    │   ├── raga.py           # Indian Classical rules (Arohana/Avarohana)
    │   ├── maqam.py          # Arabic Maqam & Iqa inputs
    │   └── gamelan.py        # Pelog/Slendro tuning & cyclic structures
    ├── generators/           # Algorithmic creation tools
    │   ├── markov.py         # Probabilistic generation
    │   ├── counterpoint.py   # Voice-leading rules engine
    │   └── genre/            # High-level style templates
    │       ├── jazz.py
    │       ├── edm.py
    │       └── lofi.py
    └── utils/                # Analysis and Helpers
        ├── analysis.py       # Key/Chord detection
        └── constants.py      # Common lookup tables
```

### 1.2 Core Integration Strategy

*   **Composability**: All high-level objects (`Chord`, `Scale`) must allow conversion to `Value` objects or lists of `Note`s that NaSong's core engine can render.
*   **Lazy Evaluation**: Generators should be capable of lazy evaluation to support infinite streams for the realtime system.
*   **Statefulness**: While `Value` is functional, generators may need `RandomState` seeds to ensure reproducibility.

## 2. Pseudo-Realtime "Algo-Rave" System

The goal is to enable a live-coding workflow where code changes take effect in the *next* audio chunk without stopping playback.

### 2.1 Architecture

We will introduce a `LiveSession` manager that decouples the **Standard Time** (the song's progression) from the **Wall Clock Time** (playback).

#### Components:
1.  **`CodeWatcher`**: Monitors the target Python file for file system events (saves).
2.  **`SessionLoader`**: Safely reloads the user module using `importlib`. It handles syntax errors gracefully (logging them without crashing the session).
3.  **`ChunkGenerator`**:
    *   Maintains a persistent `cursor` (current sample index).
    *   Calls the *current* `song` function to generate audio for `[cursor, cursor + chunk_size]`.
    *   Advances `cursor`.
4.  **`AudioBuffer`**: A simplified ring buffer or a queue that feeds an audio output stream.
5.  **`Streamer`**: Uses a library like `PyAudio`, `sounddevice`, or pipes data to `ffplay`/`mpv` stdin to play audio.

### 2.2 The "Hot-Swap" Logic

When `CodeWatcher` detects a change:
1.  **Parse & Verify**: Basic syntax check.
2.  **Reload**: `importlib.reload(user_module)`.
3.  **Update**: The `LiveSession` swaps its reference to `user_module.song`.
4.  **Next Chunk**: The next call to `ChunkGenerator` uses the *new* function `f_new(t)` but with the *continued* time range `t`.

#### Continuity Handling
*   **Phase Continuity**: Since NaSong is functional (`y = f(t)`), swapping `f` at `t` is mathematically precise. However, sudden jumps in waveform (e.g., sine 440Hz -> sine 442Hz) causing clicks will be handled by a fast cross-fade code (e.g., 5-10ms) between the old and new chunk at the boundary.
*   **Stateful Objects**: If the user uses stateful generators (e.g., a counter), reloading the module might reset them. We can introduce a `State` dictionary passed to the `song` function that persists across reloads.

### 2.3 Proposed CLI Command

`nasong-live my_song.py --chunk-size 0.5 --crossfade 0.01`

## 3. Implementation Roadmap

### Phase 1: Core Theory Foundation
*   [ ] Create `src/nasong/theory/` directory structure.
*   [ ] Implement `Note`, `Interval`, `Scale` core classes.
*   [ ] Implement `Western` system defaults (Major/Minor).

### Phase 2: Fundamental Generators
*   [ ] Implement `Chord` and `ChordProgression`.
*   [ ] Implement basic `Rhythm` generators.
*   [ ] Create adapter: `nasong.theory.to_value(theory_obj) -> Value`.

### Phase 3: "NaSL" (NaSong Language) DSL
*   [ ] Implement `nasong.dsl` module.
    *   [ ] `units.py`: Helper classes for `Hz`, `BPM`, `Ms`.
    *   [ ] `chain.py`: Operator overloading for `>>` support in `Value`.
    *   [ ] `decorators.py`: `@instrument` and `@effect` wrappers.
*   [ ] Implement "Stateless Generators" for rhythm/sequencing that depend on `t` rather than internal counters (crucial for hot-swapping).

### Phase 4: The "Algo-Rave" Engine
*   [ ] Research/Prototype audio streaming (recommend `sounddevice` or `pyaudio` as optional dependency).
*   [ ] Implement `CodeWatcher` and `SessionLoader`.
*   [ ] Build `LiveSession` loop with chunking.
*   [ ] Update `nasong-live` CLI to support DSL context.

### Phase 5: Advanced Systems & Styles
*   [ ] Implement non-western systems (Raga/Maqam placeholders).
*   [ ] Build `Jazz` and `EDM` style generators.
