# NaSong DSL & Advanced Usage Research

## 1. Concept: "NaSL" (NaSong Language) - A Python Internal DSL

To satisfy the need for a "descriptive / programming language" that is "easily compatible," we propose a **Python Internal DSL**. This uses standard Python syntax but leverages operator overloading, decorators, and factory functions to create a concise, expressive dialect optimized for musical composition and live coding (algorave).

### 1.1 Core Syntax Philosophy
1.  **Conciseness**: Minimize boilerplate. `Frequency(440)` -> `Hz(440)` or just `440`.
2.  **Piping**: Use `>>` or `.chain()` for signal flow (like tying guitar pedals).
3.  **Time as Rhythm**: String-based rhythm definitions (e.g., `"x--x"`).
4.  **Contexts**: Use `with` blocks for scales, keys, and tempos.

## 2. Examples of Advanced Usage

### 2.1 The "Neuromancer" Bass (Advanced Instrument)

**Objective**: Create a heavy, darker neuro-bass with FM synthesis, detuning, and filter modulation.

**Standard NaSong (Current Style - Verbose):**
```python
def neuromancer_bass(freq: Value, gate: Value) -> Value:
    # Operators
    op1 = Osc.Sine(freq)
    op2 = Osc.Sine(freq * 2.01) # Slight detune

    # FM Modulation
    modulator = Osc.Sine(freq * 0.5) * Env.ADSR(gate, 0.01, 0.2, 0, 0) * 500
    carrier = Osc.Saw(freq + modulator)

    # Filter
    cutoff = Env.ADSR(gate, 0.1, 0.5, 0.2, 1.0) * 2000 + 100
    filtered = Filter.LowPass(carrier, cutoff)

    # Amp
    amp = Env.ADSR(gate, 0.05, 0.2, 0.8, 0.3)
    return filtered * amp
```

**Proposed NaSL (DSL Style - Fluent):**
```python
@instrument
def neuromancer_bass(f, g):
    # Parameter inputs automatically wrapped
    mod = Sine(f * 0.5) * ADSR(g, 0.01, 0.2) * 500

    src = (
        Saw(f + mod)                 # Carrier with FM
        + Sine(f * 2.01) * 0.5       # Sub-oscillator
    )

    # Signal Chain: Source -> Distortion -> Filter -> Amp
    return src >> Distort.Tanh(2.0) \
               >> LPF(cutoff=ADSR(g, 0.1, 0.5, 0.2) * 2000 + 100) \
               >> Amp(ADSR(g, 0.05, 0.2, 0.8, 0.3))
```

### 2.2 The "Algo-Rave" Set (Live Coding Session)

**Objective**: A live loop with drums, bass, and pads that changes over time.

```python
# live_set.py

# Global Session Context (BPM, Key)
with Session(bpm=135, key="C#", scale="minor"):

    # 1. Drums: Pattern-based sequencing
    kick = Kick909() \
           .trig("x---x---x---x---")

    hats = HiHat() \
           .trig("x-x-x-x-xxx-x-x-") \
           .humanize(amount=0.05) # Add jitter

    # 2. Bass: Generative Euclidian Rhythm
    bass_seq = Seq.Euclidean(steps=16, hits=5)
    bass = MonoSynth(wave="saw") \
           .play(notes="I I IV V", rhythm=bass_seq) \
           .sidechain(source=kick) # Audio ducking

    # 3. Lead: Stochastic/Generative
    lead = Pluck() \
           .arpeggiate(chord="Im9", pattern="up-down") \
           .prob(0.8) \
           .reverb(mix=0.4)

    # Output Mix
    song = Mix([kick, hats, bass, lead]) >> Limiter()

```

### 2.3 "Cinematic Drone" (Generative Song File)

**Objective**: A slowly evolving texture using long time-scales.

```python
def song(t):
    with Scale("D", "dorian"):
        # Create 3 slow layers
        layer1 = Pad().play(chord="I", duration=10)
        layer2 = Pad().play(chord="IV", duration=10).delay(5)

        # Slow LFO controlling filter globally
        global_movement = LFO.Sine(freq=0.05)

        texture = (layer1 + layer2) >> HPF(cutoff=global_movement.map(100, 500))

        return texture
```

## 3. Potential Constraints & Challenges identified from these examples

1.  **State Management (The "Click" Problem)**:
    *   *Issue*: In the Algo-Rave example, if I change the bass rhythm from `hits=5` to `hits=6` and reload the code *while* the song is at bar 3, beat 2... what happens?
    *   *Risk*: If the new generator resets to default (pulse 0), the bass will jump out of sync with the drums.
    *   *Solution needed*: The `Seq` generators must be "stateless functions of global time" (derived from `t`) OR the system needs to persist state maps key-ed by variable name (fragile).

2.  **Resource Heavy Graphs**:
    *   *Issue*: `>> Reverb()` implies a complex convolution or feedback delay network. Doing this per-voice in `neuromancer_bass` (if it were polyphonic) kills CPU.
    *   *Solution*: Distinguish between *Voice-Level* graph (per note) and *Bus-Level* graph (global effects). The DSL must make this distinction clear (e.g., `Voice()` vs `Mix()`).

3.  **Python Overhead**:
    *   *Issue*: Calling `Note.freq` inside a sample loop in Python 44100 times a second is impossible.
    *   *Solution*: All DSL constructs must compile down to the **vectorized** `Value` graph (NumPy/Torch) at initialization time, NOT run at render time.

4.  **String Parsing Overhead**:
    *   *Issue*: Parsing `"x---x---"` every frame is bad.
    *   *Solution*: Pre-parsing caching.

## 4. Proposed DSL Modules

*   **`nasong.dsl.units`**: `Hz`, `BPM`, `Ms`, `Db`
*   **`nasong.dsl.patterns`**: `P('x---')`, `Euclid(16, 5)`
*   **`nasong.dsl.effects`**: `Chain`, `Mix`, `Sidechain`
*   **`nasong.dsl.live`**: `Session`, `Clock`, `Reload`
