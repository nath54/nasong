# The Value Architecture

This document explains the core engine of NaSong — the `Value` system — and how all audio is represented, composed, and rendered.

---

## Philosophy: Everything is a Value

In NaSong, every signal — oscillators, envelopes, modulation, noise, even constants — is a subclass of `Value`. A `Value` is a **function of discrete sample indexes**: given an array of time-based indexes, it returns an array of audio samples.

This functional approach enables:
- **Composition** — Values can be nested, summed, multiplied, and chained
- **Lazy evaluation** — The audio graph is constructed declaratively, then rendered in chunks
- **Dual backend** — The same graph runs on NumPy (CPU) or PyTorch (GPU/autograd)

---

## The `Value` Base Class

```
nasong.core.value.Value
```

All audio nodes inherit from this abstract class. It defines the interface that every node must implement.

### Core Methods

| Method | Backend | Signature | Purpose |
| :--- | :--- | :--- | :--- |
| `get_item` | Scalar | `(index: int, sample_rate: int) → float` | Single-sample fallback |
| `getitem_np` | NumPy | `(indexes_buffer: NDArray, sample_rate: int) → NDArray` | Vectorized CPU rendering |
| `getitem_torch` | PyTorch | `(indexes_buffer: Tensor, sample_rate: int, device) → Tensor` | GPU rendering + autograd |

In practice, `getitem_np` is the workhorse for standard audio generation. `getitem_torch` is used for training and hardware-accelerated rendering.

### Operator Overloads

`Value` supports Python arithmetic for concise graph construction:

| Operator | Result |
| :--- | :--- |
| `a + b` | `Sum([a, b])` |
| `a * b` | `Product(a, b)` |
| `a - b` | `a + (b * -1)` |
| `a / b` | `a * (1/b)` |
| `a % b` | `Modulo(a, b)` |
| `a ** b` | `Pow(a, b)` |

---

## Value Categories

### Basic Values

Simple building blocks:

| Class | Description | Example |
| :--- | :--- | :--- |
| `Constant(v)` | A fixed value | `Constant(440.0)` |
| `c(v)` | Alias for `Constant` | `c(0.5)` |
| `Identity()` | Returns the sample index itself | Used as the "time" input |
| `WhiteNoise()` | Random values in [-1, 1] per sample | Percussion, textures |
| `RandomChoice(values)` | Picks a random value from a list | |
| `RandomFloat(lo, hi)` | Random float in a range | |
| `RandomInt(lo, hi)` | Random int in a range | |

### Complex / Waveform Values

Oscillators and envelope generators:

| Class | Description | Key Parameters |
| :--- | :--- | :--- |
| `Sin(value, frequency, amplitude, delta)` | Sine oscillator | `frequency` in **rad/s** |
| `Cos(value, frequency, amplitude, delta)` | Cosine oscillator | `frequency` in **rad/s** |
| `Sawtooth(value, frequency, amplitude, delta)` | Sawtooth wave | `frequency` in **Hz** |
| `Square(value, frequency, amplitude, delta)` | Square wave | `frequency` in **Hz** |
| `Triangle(value, frequency, amplitude, delta)` | Triangle wave | `frequency` in **Hz** |
| `BandLimitedSawtooth(...)` | Anti-aliased sawtooth | |
| `BandLimitedSquare(...)` | Anti-aliased square | |
| `ADSR2(time, note_start, note_duration, ...)` | One-shot ADSR envelope | attack, decay, sustain, release |
| `ExponentialADSR(...)` | Exponential-curve ADSR | |
| `ExponentialDecay(...)` | Simple exponential decay | |
| `Distortion(value, amount)` | Waveshaping distortion | |
| `Log(value)` | Natural logarithm | |
| `Pow(value, exponent)` | Power function | |

> [!NOTE]
> `Sin` and `Cos` expect frequency in **radians per second** (Hz × 2π).
> `Sawtooth`, `Square`, and `Triangle` expect frequency in **Hz**.

### Multi-Item Operations

Combine multiple Value nodes:

| Class | Description | Example |
| :--- | :--- | :--- |
| `Sum(values)` | Additive mixing | `Sum([osc1, osc2, osc3])` |
| `Product(a, b)` | Multiply two values | `Product(osc, envelope)` |
| `PonderedSum(values, weights)` | Weighted mix | |
| `Max(values)` | Element-wise maximum | |
| `Min(values)` | Element-wise minimum | |

### Single-Item Operations

Transform a single Value:

| Class | Description |
| :--- | :--- |
| `Abs(value)` | Absolute value |
| `BasicScaling(value, mult_scale, sum_scale)` | `value * mult + sum` |
| `Clamp(value, min, max)` | Clip to range |
| `HighPass(value, ...)` | High-pass filter |
| `LowPass(value, ...)` | Low-pass filter |
| `MaskTreshold(value, threshold)` | Binary threshold mask |
| `Modulo(value, divisor)` | Modulo operation |
| `Polynom(value, coefficients)` | Polynomial evaluation |
| `Sequencer(time, instrument_factory, note_data_list)` | Note scheduling |
| `TimeInterval(value, start, end)` | Gate: active only in [start, end] |

### Standalone Utilities

| Function / Class | Description |
| :--- | :--- |
| `generate_harmonics(time, base_freq, n, falloff, sr)` | Band-limited harmonic series |
| `LFO(time, rate_hz, waveform_class, amplitude, delta)` | Low-Frequency Oscillator helper |
| `Formant(...)` | Vocal formant filter |
| `generate_formant_harmonics(...)` | Formant harmonic series |
| `SimpleMelody(...)` | Simple melodic sequence |
| `midi_note_to_freq(midi)` | MIDI → Hz conversion |
| `get_chord_frequencies(chord)` | Chord → frequency list |
| `input_args_to_values(...)` | Convert raw args to Value nodes |

### Training Values

| Class | Description |
| :--- | :--- |
| `ValueTrainableParameter(name, initial)` | An optimizable parameter for gradient-based training |

---

## Creating a New Value Node

Follow these steps to add a custom Value node to NaSong:

### 1. Create the file

Create `src/nasong/core/values/<category>/value_my_node.py`

### 2. Implement the class

```python
import numpy as np
from numpy.typing import NDArray
from nasong.core.value import Value

class MyNode(Value):
    """Description of what this node does."""

    def __init__(self, value: Value, my_param: float) -> None:
        super().__init__()
        self.value = value
        self.my_param = my_param

    def get_item(self, index: int, sample_rate: int) -> float:
        x = self.value.get_item(index, sample_rate)
        return x * self.my_param  # Your transform

    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        x = self.value.getitem_np(indexes_buffer, sample_rate)
        return x * self.my_param  # Vectorized version
```

### 3. Register in `all_values.py`

Add the import and `__all__` entry in `src/nasong/core/all_values.py`:

```python
from nasong.core.values.<category>.value_my_node import MyNode
# Add "MyNode" to __all__
```

### 4. (Optional) Add PyTorch support

Override `getitem_torch` for GPU acceleration and gradient support.

### 5. (Optional) Add backward support

Override `backward` for custom gradient computation in the NumPy training engine.

---

## The Rendering Pipeline

```
User Script
    │
    │  song(time) → Value graph
    ▼
┌──────────────────┐
│  Identity()       │  ← The "clock": returns sample index
│  × (1 / SR)      │  ← BasicScaling converts to seconds
│  = time Value     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Value Graph      │  ← Tree of nested Value nodes
│  (Sin, Sum, ...)  │
└────────┬─────────┘
         │
         │  getitem_np(indexes_buffer, sample_rate)
         ▼
┌──────────────────┐
│  Audio Buffer     │  ← NDArray[np.float32]
│  (chunk of N      │
│   samples)        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  WAV Export       │  ← Normalize + int16 conversion
└──────────────────┘
```

### Chunk-Based Rendering

For live playback, the `RenderEngine` renders audio in fixed-size chunks (default 4096 samples) and caches them. Chunks near the playback cursor are prioritized.

---

## NumPy Vectorization

`getitem_np` receives an entire buffer of sample indexes and must return an array of the same shape. This enables NumPy's vectorized operations for efficient batch processing:

```python
def getitem_np(self, indexes_buffer, sample_rate):
    # Evaluate child nodes
    x = self.child.getitem_np(indexes_buffer, sample_rate)
    # Apply vectorized transform
    return np.clip(x * self.gain, -1.0, 1.0)
```

---

## PyTorch Path

`getitem_torch` follows the same pattern but uses PyTorch tensors. This enables:
- **GPU acceleration** for faster rendering
- **Automatic differentiation** for gradient-based optimization of `ValueTrainableParameter` nodes

```python
def getitem_torch(self, indexes_buffer, sample_rate, device="cpu"):
    x = self.child.getitem_torch(indexes_buffer, sample_rate, device)
    return torch.clip(x * self.gain, -1.0, 1.0)
```

---

## The `backward()` Method

For the NumPy training engine (non-PyTorch), Value nodes can implement manual gradient computation:

```python
def backward(self, grad_output, context, sample_rate):
    """
    Args:
        grad_output: Gradient flowing back from downstream nodes.
        context: Dictionary storing intermediate values from the forward pass.
        sample_rate: Audio sample rate.
    """
    # Propagate gradients to child nodes
    self.child.backward(grad_output * self.gain, context, sample_rate)
```

This enables training without PyTorch, using a custom autograd engine.

---

## `ParameterContext` — Managing Trainable Parameters

The `ParameterContext` is a context manager for capturing or injecting `ValueTrainableParameter` values:

```python
from nasong.core.value import ParameterContext

# Capture mode: collect all parameter names and initial values
with ParameterContext(capture=True) as ctx:
    graph = build_instrument(time)
params = ctx.captured_params

# Injection mode: set specific parameter values
with ParameterContext(parameters={"vibrato_depth": 0.02}):
    graph = build_instrument(time)
```

---

## Related Documentation

- [Core Values API](api/core_values.md) — Complete reference for all Value nodes
- [Architecture](architecture.md) — Project-wide module diagram
- [Contributing](contributing.md) — How to add new Value nodes step by step
