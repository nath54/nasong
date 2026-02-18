# Value Nodes Reference

Complete reference for all `Value` nodes available via `nasong.core.all_values`.

```python
from nasong.core.all_values import ...
```

But you can also import them from their own modules, like:

```python
from nasong.core.values.basic.value_constant import Constant
```

---

## Basic Values

### `Constant(value)`

Returns a fixed value for every sample.

```python
volume = Constant(0.5)
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `value` | `float` | The constant value to return |

The alias `c(value)` is also available.

---

### `Identity()`

Returns the raw sample index. Used as the "clock" or "time" input.

```python
time = Identity()
```

No parameters.

---

### `WhiteNoise()`

Returns uniformly distributed random values in [-1, 1] per sample.

```python
noise = WhiteNoise()
```

No parameters.

---

### `RandomChoice(values)`

Returns a randomly selected value from the provided list (fixed at construction time).

---

### `RandomFloat(lo, hi)`

Returns a random float in `[lo, hi]` (fixed at construction time).

---

### `RandomInt(lo, hi)`

Returns a random integer in `[lo, hi]` (fixed at construction time).

---

## Oscillators

### `Sin(value, frequency, amplitude=1.0, delta=0.0)`

Sine wave oscillator.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `value` | `Value` | Time input |
| `frequency` | `Value` | Frequency in **radians/sec** (Hz × 2π) |
| `amplitude` | `Value` | Output amplitude. Default: `Constant(1.0)` |
| `delta` | `Value` | Phase offset. Default: `Constant(0.0)` |

```python
# 440 Hz sine wave
osc = Sin(time, Constant(440.0 * 6.2832))
```

> [!IMPORTANT]
> `Sin` and `Cos` use **radians per second**, not Hz. Multiply Hz by `2π ≈ 6.2832`.

---

### `Cos(value, frequency, amplitude=1.0, delta=0.0)`

Cosine wave oscillator. Same parameters and conventions as `Sin`.

---

### `Sawtooth(value, frequency, amplitude=1.0, delta=0.0)`

Sawtooth wave oscillator.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `value` | `Value` | Time input |
| `frequency` | `Value` | Frequency in **Hz** |
| `amplitude` | `Value` | Output amplitude |
| `delta` | `Value` | Phase offset |

```python
saw = Sawtooth(time, Constant(220.0))
```

---

### `Square(value, frequency, amplitude=1.0, delta=0.0)`

Square wave oscillator. Frequency in **Hz**.

---

### `Triangle(value, frequency, amplitude=1.0, delta=0.0)`

Triangle wave oscillator. Frequency in **Hz**.

---

### `BandLimitedSawtooth(...)`

Anti-aliased sawtooth using additive synthesis (sum of harmonics below Nyquist).

---

### `BandLimitedSquare(...)`

Anti-aliased square wave using additive synthesis.

---

## Envelopes

### `ADSR2(time, note_start, note_duration, attack_time, decay_time, sustain_level, release_time)`

One-shot Attack-Decay-Sustain-Release envelope.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `time` | `Value` | Global time input |
| `note_start` | `float` | Note onset time (seconds) |
| `note_duration` | `float` | Note duration (seconds) |
| `attack_time` | `float` | Attack duration (seconds) |
| `decay_time` | `float` | Decay duration (seconds) |
| `sustain_level` | `float` | Sustain amplitude (0.0–1.0) |
| `release_time` | `float` | Release duration (seconds) |

```python
env = ADSR2(time, 0.0, 2.0, 0.05, 0.1, 0.7, 0.3)
```

---

### `ExponentialADSR(...)`

ADSR envelope with exponential curves for more natural dynamics.

---

### `ExponentialDecay(...)`

Simple exponential decay curve.

---

## Math & Transfer Functions

### `Distortion(value, amount)`

Waveshaping distortion via soft clipping.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `value` | `Value` | Input signal |
| `amount` | `float` | Distortion intensity |

---

### `Log(value)`

Natural logarithm of the input.

---

### `Pow(value, exponent)`

Raises the input to a power.

---

## Multi-Item Operations

### `Sum(values)`

Additive mix of multiple values.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `values` | `list[Value]` | List of values to sum |

```python
mix = Sum([osc1, osc2, osc3])
# Equivalent: osc1 + osc2 + osc3
```

---

### `Product(a, b)` or `Product(values)`

Multiply two values (amplitude modulation, envelope application).

```python
signal = Product(osc, envelope)
# Equivalent: osc * envelope
```

---

### `PonderedSum(values, weights)`

Weighted sum of multiple values.

---

### `Max(values)` / `Min(values)`

Element-wise maximum / minimum of multiple values.

---

## Single-Item Operations

### `BasicScaling(value, mult_scale, sum_scale)`

Linear transform: `output = value × mult_scale + sum_scale`

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `value` | `Value` | Input |
| `mult_scale` | `Value` | Multiplicative factor |
| `sum_scale` | `Value` | Additive offset |

---

### `Abs(value)`

Absolute value. Useful for full-wave rectification.

---

### `Clamp(value, min, max)`

Clips the signal to `[min, max]`.

---

### `HighPass(value, ...)`

High-pass filter.

---

### `LowPass(value, ...)`

Low-pass filter.

---

### `MaskTreshold(value, threshold)`

Binary mask: outputs 1.0 where `value > threshold`, 0.0 otherwise.

---

### `Modulo(value, divisor)`

Modulo operation. Used for looping patterns: `time % loop_duration`.

```python
loop_time = Modulo(time, Constant(4.0))
# Equivalent: time % Constant(4.0)
```

---

### `Polynom(value, coefficients)`

Evaluates a polynomial on the input signal.

---

### `Sequencer(time, instrument_factory, note_data_list)`

Schedules and renders multiple notes using an instrument factory.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `time` | `Value` | Global time input |
| `instrument_factory` | `Callable` | `(time, freq, start, dur, vel) → Value` |
| `note_data_list` | `list[tuple]` | `[(freq, start, dur, vel), ...]` |

```python
notes = [(440.0, 0.0, 1.0, 0.8), (554.37, 1.0, 1.0, 0.7)]
seq = Sequencer(time, instrument_factory=my_synth, note_data_list=notes)
```

---

### `TimeInterval(value, start, end)`

Gates the signal: active only when time is in `[start, end]`.

---

## Standalone Utilities

### `generate_harmonics(time, base_frequency, num_harmonics, amplitude_falloff, sample_rate, base_amplitude=1.0)`

Creates a band-limited sum of sine harmonics (prevents aliasing above Nyquist).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `time` | `Value` | Time input |
| `base_frequency` | `float` | Fundamental frequency (Hz) |
| `num_harmonics` | `int` | Max harmonics to generate |
| `amplitude_falloff` | `float` | Amplitude multiplier per harmonic |
| `sample_rate` | `int` | Audio sample rate |
| `base_amplitude` | `Value` | Overall amplitude |

---

### `LFO(time, rate_hz, waveform_class, amplitude=1.0, delta=0.0)`

Helper to create Low-Frequency Oscillators. Automatically handles the Hz-to-rad/s conversion for `Sin`/`Cos`.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `time` | `Value` | Time input |
| `rate_hz` | `Value` | LFO frequency in Hz |
| `waveform_class` | `type` | `Sin`, `Triangle`, `Square`, etc. |
| `amplitude` | `Value` | LFO amplitude |
| `delta` | `Value` | Phase offset |

```python
from nasong.core.all_values import LFO, Sin, Constant, Identity

vibrato = LFO(Identity(), Constant(5.0), Sin, amplitude=Constant(0.02))
```

---

### `Formant(...)` / `generate_formant_harmonics(...)`

Vocal formant synthesis.

---

### `SimpleMelody(...)`

Generates a simple melodic sequence as a value.

---

### `midi_note_to_freq(midi_note)`

Converts a MIDI note number to frequency in Hz.

---

### `get_chord_frequencies(chord)`

Returns a list of frequencies for a given chord.

---

### `input_args_to_values(...)`

Converts raw arguments (floats, ints) into `Value` objects.

---

## Training

### `ValueTrainableParameter(name, initial_value)`

An optimizable scalar parameter. Used in the training pipeline with `ParameterContext`.

```python
from nasong.core.value import ValueTrainableParameter

depth = ValueTrainableParameter("vibrato_depth", 0.01)
```

---

## Related Documentation

- [Value System](../value_system.md) — Architecture deep dive
- [Effects Reference](effects.md) — Pre-built effects
- [Instruments](../instruments.md) — Pre-built instrument library
