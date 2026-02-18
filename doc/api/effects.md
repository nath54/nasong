# Effects Reference

Pre-built effect generators available in `nasong.instruments.effects` and `nasong.core.values`.

---

## `ADSR_Piano` — Looping Envelope

A looping Attack-Decay-Sustain-Release envelope. Unlike `ADSR2` (which is one-shot), `ADSR_Piano` uses modulo arithmetic on time, causing the envelope to repeat.

```python
from nasong.instruments.effects import ADSR_Piano

env = ADSR_Piano(
    time=time,
    note_freq=440.0,
    attack=0.05,
    decay=0.1,
    sustain_level=0.7,
    release=0.3,
    note_duration=1.0,
)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `time` | `Value` | — | Time provider |
| `note_freq` | `float` | — | Target frequency (currently unused) |
| `attack` | `float` | `0.05` | Attack duration (seconds) |
| `decay` | `float` | `0.1` | Decay duration (seconds) |
| `sustain_level` | `float` | `0.7` | Sustain amplitude (0.0–1.0) |
| `release` | `float` | `0.3` | Release duration (seconds) |
| `note_duration` | `float` | `1.0` | Total note cycle duration (seconds) |

> [!WARNING]
> This envelope **loops continuously**. For one-shot note envelopes, use `ADSR2` instead.

### Usage Example

```python
from nasong.core.all_values import Sin, Constant, Identity
from nasong.instruments.effects import ADSR_Piano

time = Identity()
osc = Sin(time, Constant(440.0 * 6.2832))
env = ADSR_Piano(time, 440.0, attack=0.02, sustain_level=0.6, note_duration=0.5)
signal = osc * env * 0.3
```

---

## `Vibrato` — Frequency Modulation

Generates a frequency modulation signal for vibrato effects. This is **not** an audio-input effect — it is a Value generator that produces a modulated frequency curve.

```python
from nasong.instruments.effects import Vibrato

freq_with_vibrato = Vibrato(
    time=time,
    base_frequency=440.0,
    vibrato_rate=5.0,
    vibrato_depth=0.015,
)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `time` | `Value` | — | Global time input |
| `base_frequency` | `float` | — | Center frequency (Hz) |
| `vibrato_rate` | `float` | `5.0` | LFO speed (Hz) |
| `vibrato_depth` | `float` | `0.015` | Modulation amount (fraction, e.g., 0.015 = 1.5%) |

### How It Works

The output is a **frequency** value (not audio). Feed it into an oscillator as the frequency parameter:

```python
freq = Vibrato(time, base_frequency=440.0, vibrato_rate=6.0, vibrato_depth=0.02)
osc = Sin(time, freq * Constant(6.2832))  # Convert to rad/s
```

### Usage Example — Violin Vibrato

```python
from nasong.core.all_values import Sin, ADSR2, Constant, Identity
from nasong.instruments.effects import Vibrato

time = Identity()

# Vibrato: gently wobbles the pitch at 5 Hz, ±2% depth
freq = Vibrato(time, 440.0, vibrato_rate=5.0, vibrato_depth=0.02)

# Oscillator using the modulated frequency
osc = Sin(time, freq * Constant(6.2832))

# Envelope
env = ADSR2(time, 0.0, 3.0, 0.1, 0.2, 0.7, 0.5)

signal = osc * env * 0.4
```

---

## `Distortion` — Waveshaping

Soft-clipping distortion via the `tanh` waveshaper.

```python
from nasong.core.all_values import Distortion

distorted = Distortion(value=clean_signal, amount=5.0)
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `value` | `Value` | Input audio signal |
| `amount` | `float` | Distortion drive (higher = more clipping) |

---

## `BandLimitedSawtooth` — Anti-Aliased Sawtooth

A sawtooth wave built from an additive series of sine harmonics, stopping at the Nyquist frequency to prevent aliasing.

```python
from nasong.core.all_values import BandLimitedSawtooth, Constant

saw = BandLimitedSawtooth(value=time, frequency=Constant(220.0))
```

---

## `BandLimitedSquare` — Anti-Aliased Square

A square wave built from odd harmonics only (1, 3, 5, …), band-limited to prevent aliasing.

```python
from nasong.core.all_values import BandLimitedSquare, Constant

sq = BandLimitedSquare(value=time, frequency=Constant(220.0))
```

---

## `LFO` — Low-Frequency Oscillator

A helper function that simplifies LFO creation by handling the Hz-to-rad/s conversion for `Sin`/`Cos` automatically.

```python
from nasong.core.all_values import LFO, Sin, Triangle, Constant, Identity

# Sine LFO at 3 Hz
lfo = LFO(Identity(), Constant(3.0), Sin, amplitude=Constant(0.5))

# Triangle LFO at 1 Hz
lfo = LFO(Identity(), Constant(1.0), Triangle, amplitude=Constant(0.3))
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `time` | `Value` | — | Time input |
| `rate_hz` | `Value` | — | LFO frequency in Hz |
| `waveform_class` | `type` | — | `Sin`, `Cos`, `Triangle`, `Square`, etc. |
| `amplitude` | `Value` | `Constant(1.0)` | Output amplitude |
| `delta` | `Value` | `Constant(0.0)` | Phase offset |

### Usage Example — Tremolo

```python
# Apply a tremolo (amplitude modulation) at 4 Hz
lfo = LFO(Identity(), Constant(4.0), Sin, amplitude=Constant(0.3))
modulated = osc * (Constant(0.7) + lfo)  # Volume varies between 0.4 and 1.0
```

---

## `generate_harmonics` — Harmonic Series

Generates a band-limited sum of harmonic sine waves with anti-aliasing.

```python
from nasong.core.all_values import generate_harmonics

rich_tone = generate_harmonics(
    time=time,
    base_frequency=220.0,
    num_harmonics=16,
    amplitude_falloff=0.5,
    sample_rate=44100,
)
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `time` | `Value` | Time input |
| `base_frequency` | `float` | Fundamental frequency (Hz) |
| `num_harmonics` | `int` | Maximum number of harmonics |
| `amplitude_falloff` | `float` | Each harmonic's amplitude = previous × falloff |
| `sample_rate` | `int` | Audio sample rate (for Nyquist check) |
| `base_amplitude` | `Value` | Overall amplitude scaling |

---

## Related Documentation

- [Core Values API](core_values.md) — All Value nodes
- [Instruments](../instruments.md) — Pre-built instrument library
- [Value System](../value_system.md) — Architecture of the Value engine
