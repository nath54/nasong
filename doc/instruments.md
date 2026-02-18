# Instruments Documentation

NaSong ships with two families of instruments: **Normal instruments** for playback and composition, and **Trainable instruments** for machine-learning-driven sound matching. This document explains both, their differences, and how to use each.

---

## Normal Instruments

Normal instruments are factory functions that produce a `Value` audio graph from a set of parameters. They are the building blocks you use in songs and live scripts.

### Sequencer Contract

When you call `render(progression, time, InstrumentFactory, bpm=120)`, the `Sequencer` internally calls your factory like this:

```python
instrument_factory(time, frequency, start_time, duration, velocity)
```

Every instrument that can be used with `render()` / `Sequencer` **must** accept these 5 positional arguments:

| Argument | Type | Description |
| :--- | :--- | :--- |
| `time` | `lv.Value` | The global time signal. |
| `frequency` | `float` | Note frequency in Hz (e.g., 440.0 for A4). |
| `start_time` | `float` | When the note begins, in seconds. |
| `duration` | `float` | Length of the note in seconds. |
| `velocity` | `float` | Volume / intensity (0.0 to 1.0, sometimes higher). |

The 4th and 5th arguments (`duration`, `velocity`) should have default values so the instrument can also be used standalone.

### About `float` vs `lv.Value` for Frequency

Normal instruments type-hint `frequency` as `float`. This is because they do **inline Python float arithmetic** to build harmonics. For example, `PianoNote` does:

```python
lv.Sin(time, frequency=lv.Constant(frequency * 2 * pi2), ...)
#                                   ^^^^^^^^^^^^^^^^^^^^^^
#                                   This is float * float * float
```

If you passed an `lv.Value` instead of a `float`, the Python `*` operator would raise a `TypeError` because `lv.Value * float` doesn't produce a plain `float` — it produces another `Value` node.

**However**, some instruments like `SynthLead` wrap the frequency early:

```python
base_freq: lv.Value = lv.Sum(lv.c(frequency), vibrato_lfo)
osc = lv.BandLimitedSawtooth(time, frequency=base_freq, num_harmonics=30)
```

Here, `frequency` is wrapped into `lv.c(frequency)` immediately, and all subsequent operations use `lv.Value` arithmetic (`lv.Product`, `lv.Sum`). This instrument *could* theoretically accept an `lv.Value` for deeper modulation effects (e.g., pitch glides, portamento), but would need a small refactor: replace `lv.c(frequency)` with a direct pass-through, and use `lv.Product(frequency, lv.c(0.00005))` instead of `lv.c(0.00005 * frequency)`.

**In summary:**
- **Currently**: Normal instruments require `frequency: float`. Passing `lv.Value` will crash.
- **Why**: Inline float math like `frequency * 2 * pi2` is incompatible with `lv.Value`.
- **Potential**: Instruments that wrap frequency early (e.g., `SynthLead`) could be refactored to accept `lv.Value`, enabling continuous pitch modulation, vibrato depth control, pitch bends, etc.
- **Trainable instruments** already use `frequency: lv.Value` — they avoid float math and use `lv.Product` / `lv.Sum` everywhere.

### Available Instruments

#### Synths (`nasong.instruments.synth`)

| Instrument | Description | Extra Defaults |
| :--- | :--- | :--- |
| `SynthLead` | Thick unison lead (3 detuned sawtooths + vibrato). | `duration=1.0`, `velocity=1.0` |
| `SynthBass` | Punchy sub-bass (square + sub-octave sine). | `duration=0.5`, `velocity=1.0` |
| `SynthPad` | Atmospheric pad (3 detuned sines, slow attack). | `duration=4.0`, `velocity=1.0` |

#### Keyboards (`nasong.instruments.keyboards`)

| Instrument | Description | Extra Defaults |
| :--- | :--- | :--- |
| `PianoNote` | Acoustic piano (4 harmonics + ADSR2 envelope). | `amplitude=0.3` |
| `PianoNote2` | Alternate piano (5 harmonics + exponential ADSR). | `duration=2.0`, `amplitude=0.3` |

#### Percussion (`nasong.instruments.percussion`)

| Instrument | Description | Extra Defaults |
| :--- | :--- | :--- |
| `KickDrum` | Synthesized kick (pitch-modulated sine + click). | `duration=0.5`, `velocity=1.0` |
| `KickDrum2` | Punchier kick (steeper pitch drop + noise click). | `duration=0.5`, `velocity=1.0` |
| `Snare` | Snare drum (sine body + white noise snap). | `duration=0.3`, `velocity=1.0` |
| `SnareDrum` | Tighter, shorter snare variant. | `duration=0.2`, `velocity=1.0` |
| `HiHat` | Hi-hat (multi-sine + noise, open/closed). | `duration=0.1`, `velocity=1.0`, `open=False` |
| `CrashCymbal` | Crash cymbal (high-freq additive + random phases). | `duration=2.0`, `velocity=1.0` |

#### Bass (`nasong.instruments.bass`)

| Instrument | Description | Extra Defaults |
| :--- | :--- | :--- |
| `WobbleBass` | Electronic wobble bass (sawtooth + LFO filter + distortion). | `wobble_rate=4.0`, `amplitude=0.4` |
| `DeepBass` | Atmospheric sub-bass (pure sine + exponential decay). | `duration=0.5`, `amplitude=0.4` |

#### Bowed Strings (`nasong.instruments.bowed_strings`)

| Instrument | Description | Extra Defaults |
| :--- | :--- | :--- |
| `Violin` | Formant-shaped violin (vibrato + tremolo + bow noise). | `amplitude=0.18`, `vibrato_rate=30.0` |
| `Cello` | Deeper cello (lower formants, more harmonics). | `amplitude=0.22`, `vibrato_rate=0.1` |

#### Plucked Strings (`nasong.instruments.plucked_strings`)

| Instrument | Description | Extra Defaults |
| :--- | :--- | :--- |
| `GuitarString` | Plucked guitar (6 harmonics + brightness decay). | `duration=3.0`, `brightness=1.0` |
| `GuitarString2` | Alternate guitar with subtle noise. | `amplitude=0.4` |
| `AcousticString` | Bright acoustic pluck (5 harmonics + noise). | `duration=3.0`, `amplitude=0.3`, `decay_rate=2.0` |

#### Winds (`nasong.instruments.winds`)

| Instrument | Description | Extra Defaults |
| :--- | :--- | :--- |
| `SaxophoneNote` | Saxophone (odd-heavy harmonics + vibrato + breath noise). | `velocity=1.0` |

#### Higher-Level Helpers

These are **not** Sequencer-compatible factories. They build their own internal Sequencer:

| Function | Description |
| :--- | :--- |
| `Fingerpicking(time, bass_note, chord_notes, start_time)` | Fingerpicking pattern using `AcousticString`. |
| `Strum(time, frequencies, start_time, duration)` | Strummed chord across multiple `GuitarString`s. |

### Example: Using Normal Instruments

**In a standard nasong script:**
```python
import nasong.core.all_values as lv
from nasong.instruments.synth import SynthLead

duration = 5.0

def song(time: lv.Value) -> lv.Value:
    note = SynthLead(time, frequency=440.0, start_time=0.0, duration=2.0, velocity=0.8)
    return note
```

**In a live rave script (with Sequencer):**
```python
from nasong.theory import render
from nasong.core.values.basic.value_identity import Identity
from nasong.instruments.synth import SynthLead
from nasong.theory.generators.styles.jazz import Jazz

BPM = 120
LOOP_QUARTERS = 4
LOOP_SECONDS = (LOOP_QUARTERS * 60.0) / BPM

time = Identity() % LOOP_SECONDS

prog = Jazz.generate_random_standards_progression(length=4)
sequencer = render(prog, time, SynthLead, bpm=BPM)  # SynthLead is the factory
```

---

## Trainable Instruments

Trainable instruments are **differentiable synthesis blueprints** designed for the training pipeline. Instead of hardcoded synthesis parameters, they use `ValueTrainableParameter` nodes whose values are optimized via gradient descent to match a target audio sample.

### Key Differences from Normal Instruments

| Aspect | Normal Instrument | Trainable Instrument |
| :--- | :--- | :--- |
| **Purpose** | Playback and composition | Sound matching via gradient descent |
| **`frequency` type** | `float` (plain number) | `lv.Value` (differentiable node) |
| **Synthesis params** | Hardcoded constants | `ValueTrainableParameter` (learnable) |
| **Used with** | `render()` / `Sequencer` / `song()` | `nasong-train` pipeline |
| **Params change?** | Fixed at creation time | Updated each training epoch |
| **Sequencer-compatible?** | ✅ Yes | ❌ No (different signature) |

### Why `frequency` is `lv.Value`

In the training pipeline, the frequency is extracted from the target audio using note detection and injected as a `Value` node into the graph. This keeps the entire synthesis graph differentiable — gradients can flow from the spectral loss back through the oscillators to the trainable parameters.

```python
# Normal instrument: frequency is a float, used in float math
PianoNote(time, frequency=440.0, start_time=0.0, duration=1.0)
#                 ↓
# Inside: lv.Constant(frequency * 2 * pi2) ← float * float * float

# Trainable instrument: frequency is a Value, used in Value math
TrainableSawtoothSynth(time, frequency=lv.Constant(440.0), start_time=0.0, duration=1.0)
#                               ↓
# Inside: lv.Product(frequency, lv.Constant(pi2)) ← Value * Value
```

### Available Trainable Instruments

#### Synths (`nasong.trainable.instruments.synth`)

| Instrument | Description | Trainable Parameters |
| :--- | :--- | :--- |
| `TrainableSawtoothSynth` | Band-limited sawtooth + Exponential ADSR. | attack, decay, sustain, release |
| `TrainableSquareSynth` | Band-limited square wave + ADSR. | attack, decay, sustain, release, duty |
| `TrainableSineSynth` | Pure sine wave + ADSR. | attack, decay, sustain, release |

#### Percussion (`nasong.trainable.instruments.percussion`)

| Instrument | Description | Trainable Parameters |
| :--- | :--- | :--- |
| `TrainableKick` | Kick drum (sine + pitch sweep + click). | base_freq, sweep_amt, decay, noise, click, amp |
| `TrainableSnare` | Snare (tonal sine + noise burst). | tone_freq, decay, noise, amp |
| `TrainableHiHat` | Hi-hat (noise + metallic high-freq sines). | decay, amp, bright |

#### Melodic (`nasong.trainable.instruments.melodic`)

| Instrument | Description | Trainable Parameters |
| :--- | :--- | :--- |
| `TrainablePlucked` | Plucked string (sawtooth + pluck decay). | decay, bright, attack |
| `TrainablePiano` | Piano (3 sine harmonics + ADSR). | attack, decay, sustain, release, bright |
| `TrainableBowed` | Bowed string (sawtooth + vibrato). | attack, decay, sustain, release, vib_rate, vib_depth |

#### Bass (`nasong.trainable.instruments.bass`)

| Instrument | Description | Trainable Parameters |
| :--- | :--- | :--- |
| `TrainableBass` | Bass synth (sawtooth + sub-sine + distortion). | attack, decay, sustain, release, sub, dist |

#### Atmospheric (`nasong.trainable.instruments.atmospheric`)

| Instrument | Description | Trainable Parameters |
| :--- | :--- | :--- |
| `TrainablePad` | Pad synth (3 detuned sines, slow envelope). | attack, decay, sustain, release, detune, bright |

### Training Workflow

1.  **Prepare a target audio file** (WAV format):
    ```
    example_songs_for_training/my_sample.wav
    ```

2.  **Create a training config** (YAML):
    ```yaml
    instrument_name: saw
    target_wav: example_songs_for_training/my_sample.wav
    epochs: 500
    learning_rate: 0.01
    device: cpu
    audio:
      duration: 5.0
      sample_rate: 44100
    note_detection:
      method: legacy
    spectral_loss:
      fft_sizes: [2048, 1024, 512]
    ```
    See [Training Configuration](training_config.md) for all available parameters.

3.  **Run training**:
    ```bash
    nasong-train --config training_configs/my_experiment.yaml
    ```

4.  **Monitor progress**:
    ```bash
    nasong-monitor list
    nasong-monitor show <experiment_id>
    ```

5.  **Use the trained instrument in a song**:
    ```python
    from nasong.trainable.inference import load_trained_instrument
    import nasong.core.all_values as lv

    my_instrument = load_trained_instrument("<experiment_id>")

    def song(time: lv.Value) -> lv.Value:
        return my_instrument(
            time=time,
            frequency=lv.Constant(440),
            start_time=0.0,
            duration=2.0
        )
    ```

### Creating a Custom Trainable Instrument

You can define your own trainable instrument by following this pattern:

```python
import nasong.core.all_values as lv

def MyTrainableInstrument(
    time: lv.Value,
    frequency: lv.Value,         # Must be lv.Value for differentiability
    start_time: float,
    duration: float,
    init_amplitude: float = 0.3,
    name_prefix: str = "my_inst", # Unique prefix for parameter names
) -> lv.Value:

    # 1. Define trainable parameters
    attack = lv.ValueTrainableParameter(0.01, name=f"{name_prefix}_attack")
    decay  = lv.ValueTrainableParameter(0.1,  name=f"{name_prefix}_decay")

    # 2. Build the envelope
    env = lv.ExponentialADSR(
        time, note_start=start_time, note_duration=duration,
        attack_time=float(attack.value), decay_time=float(decay.value),
        sustain_level=0.7, release_time=0.2,
    )

    # 3. Build the oscillator (use lv.Product, NOT float *)
    freq_rads = lv.Product(frequency, lv.Constant(6.283185307179586))
    osc = lv.Sin(time, freq_rads, lv.Constant(init_amplitude))

    # 4. Combine
    return lv.Product(osc, env)
```

Key rules:
- `frequency` must be `lv.Value` (not `float`).
- Use `lv.ValueTrainableParameter` for any parameter you want the optimizer to learn.
- Give each parameter a unique `name` using `name_prefix`.
- **Never** do `frequency * 2.0` — use `lv.Product(frequency, lv.c(2.0))` instead.
- Return an `lv.Value` node.

---

## The Chunk System

NaSong has two very different rendering modes depending on whether you are compiling a song to WAV or playing live audio.

### Standard NaSong (`nasong`): Single-Pass Rendering

When you run `nasong my_song.py -o output.wav`, the `Song.render()` method renders the **entire** audio in one pass:

```python
# Inside Song.render():
idx_buffer = np.arange(0, total_samples, 1, dtype=np.float32)
audio_data = audio_value.getitem_np(idx_buffer, sample_rate)
```

There is **no chunking** — the full `np.arange(0, total_samples)` is passed to the Value graph at once. This is simple and fast (NumPy vectorization), but requires the entire audio to fit in memory.

For a 60-second song at 44100 Hz, that's `60 × 44100 = 2,646,000` float32 samples ≈ 10 MB — which is fine. Even a 10-minute song is only ~100 MB.

### Algo-Rave (`nasong-rave`): Chunk-Based Streaming

Live playback cannot compute the entire audio upfront because:
1. The loop duration is unknown (it loops forever).
2. Audio must feed the sound card in real-time (~46ms latency at 2048 samples / 44100 Hz).
3. Script changes via hot-reload require re-rendering.

The system uses two components:

#### `LiveSession` — Audio I/O

- Opens a real-time audio stream via `sounddevice` (PortAudio).
- The audio callback fires every `block_size` samples (default: 2048).
- Each callback must fill the output buffer with audio data — if it takes too long, you get glitches.

#### `RenderEngine` — Background Chunk Renderer

The `RenderEngine` pre-computes audio in fixed-size **chunks** and caches them. A background thread renders chunks ahead of the playback cursor.

**How it works:**

```
 Playback cursor
       ↓
  [chunk 0][chunk 1][chunk 2][chunk 3][chunk 4] ...
  ━━━━━━━━  ━━━━━━━━  ━━━━━━━━  ━━━━━━━━  ━━━━━━━━
  cached ✓  cached ✓  playing   queued    queued
```

1. **Chunking**: Audio is divided into fixed-size blocks of `chunk_size` samples (default: 2048, configurable via `--block-size`).

2. **Priority Queue**: Chunks near the cursor get rendered first. The priority of a chunk = `|chunk_start_sample - cursor_sample|`. So the chunk you're about to hear is always rendered before chunks 30 seconds in the future.

3. **Cache**: Each rendered chunk is stored in a dictionary: `cache[start_sample] = (audio_array, version_id)`. When the audio callback needs samples, it looks up the cache. Cache hits → audio plays. Cache misses → silence.

4. **Versioning**: When the user's script changes (hot-reload), `set_sequencer()` increments `current_version_id`. Old cached chunks are **not deleted** — they become "stale" (version mismatch). The background thread re-renders stale chunks with the new sequencer, replacing them as they complete.

5. **Window**: Only chunks within a window around the cursor are queued: -5s behind to +30s ahead.

#### Audio Callback Flow

When the sound card asks for audio:

```
1. Look up cache[cursor_sample]
2. If found → copy to output buffer → advance cursor
3. If not found → output silence → advance cursor
4. Update cursor_time on RenderEngine (re-prioritizes queue)
```

### Force Re-Render (Double Buffering)

For scripts using randomness (e.g., `Jazz.generate_random_standards_progression`), you can set:

```python
FORCE_RERENDER_EVERY = 256  # Re-render every 256 chunks
```

When the background thread has rendered `N` chunks since the last re-render:

1. Increment `current_version_id` (marks all cache as stale).
2. **Do NOT clear the cache** — stale chunks keep playing.
3. Call the `rerender_callback` (re-executes the user script → new random values).
4. Re-queue chunks near cursor for re-rendering with the new sequencer.

This is **double buffering**: the old audio keeps playing while the new version is rendered in the background. Once a chunk is re-rendered, it replaces the stale entry — no silence gaps.

### Customizing Chunk Size

The chunk size (= block size) controls the trade-off between **latency** and **stability**:

| Chunk Size | Latency | CPU Load | Best For |
| :--- | :--- | :--- | :--- |
| 256 | ~6ms | Very high | Ultra-low latency monitoring |
| 512 | ~12ms | High | Low-latency live performance |
| 1024 | ~23ms | Medium | General live coding |
| **2048** (default) | **~46ms** | **Low** | **Recommended default** |
| 4096 | ~93ms | Very low | Complex patches, slow hardware |

**Via CLI:**
```bash
# Algo-Rave TUI
nasong-rave --block-size 1024

# Headless mode
nasong-rave --headless my_script.py --block-size 4096
```

Smaller chunks mean the sound card asks for audio more frequently, which requires faster rendering. If your instrument graph is complex and the background thread can't keep up, you'll hear silence gaps (cache misses). In that case, increase the chunk size.

**Note**: The chunk size also determines the smallest unit of cache invalidation. When you hot-reload a script, all chunks must be re-rendered. Smaller chunks = more chunks to re-render, but each is faster. Larger chunks = fewer to re-render, but each takes longer. In practice, the difference is negligible.
