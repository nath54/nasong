# DSL Syntax Reference

NaSong provides an optional **Domain-Specific Language (DSL)** layer that offers a more fluent, declarative syntax for building audio graphs. The DSL sits on top of the core `Value` system and provides three sub-modules: **chaining**, **decorators**, and **units**.

---

## Signal Chaining (`nasong.dsl.chain`)

The chaining DSL uses the `>>` (right-shift) operator to feed signals through processors in a pipeline style.

### Core Classes

| Class | Description |
| :--- | :--- |
| `Chainable(value)` | Wraps a `Value` for chaining via `>>` |
| `Processor` | Base class for signal processors |
| `Gain(amount)` | Amplitude scaling processor |

### Usage

```python
from nasong.dsl.chain import Chainable, Gain
from nasong.core.all_values import Sin, Constant

# Create a sine wave and apply gain
result = Chainable(Sin(Constant(440))) >> Gain(0.5)
audio_value = result.val()  # Unwrap back to a Value
```

### Chaining Rules

| Expression | Result |
| :--- | :--- |
| `Chainable(v) >> Processor(...)` | Processor is applied to `v` |
| `Chainable(v) >> 0.5` | Shorthand for `>> Gain(0.5)` |
| `result.val()` | Extracts the underlying `Value` |

### Creating Custom Processors

Subclass `Processor` and implement `__call__`:

```python
from nasong.dsl.chain import Processor
from nasong.core.value import Value
from nasong.core.all_values import LowPass

class LPF(Processor):
    """Low-pass filter processor."""

    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def __call__(self, source: Value) -> Value:
        return LowPass(source, cutoff=self.cutoff)
```

Then use it in a chain:

```python
audio = Chainable(osc) >> Gain(0.8) >> LPF(2000) >> 0.5
```

---

## Comparison: DSL vs. Python API

The DSL is syntactic sugar. Both approaches produce identical `Value` graphs:

**Python API:**
```python
from nasong.core.all_values import Sin, Constant, Product

osc = Sin(Constant(440))
scaled = Product(osc, Constant(0.5))
```

**DSL:**
```python
from nasong.dsl.chain import Chainable, Gain
from nasong.core.all_values import Sin, Constant

audio = (Chainable(Sin(Constant(440))) >> Gain(0.5)).val()
```

The DSL is most useful for **long processing chains** where the pipeline style is more readable.

---

## Decorators (`nasong.dsl.decorators`)

Registration decorators for marking functions as instruments or effects:

### `@instrument`

Marks a function as a NaSong instrument. The function should return a `Value` graph:

```python
from nasong.dsl.decorators import instrument

@instrument
def my_synth(time, freq, duration):
    osc = Sin(time, Constant(freq * 6.2832))
    env = ADSR2(time, 0.0, duration, 0.01, 0.1, 0.5, 0.2)
    return osc * env
```

### `@effect`

Marks a function as an audio effect. Effects take a source `Value` as their first argument:

```python
from nasong.dsl.decorators import effect

@effect
def chorus(source, depth=0.3, rate=1.5):
    lfo = LFO(Identity(), Constant(rate), Sin, Constant(depth))
    return source * (Constant(1.0) + lfo)
```

> [!NOTE]
> These decorators currently add a metadata flag (`_is_nasong_instrument` / `_is_nasong_effect`) for discovery by the TUI and potential future tooling. They do not alter the function's behavior.

---

## Units (`nasong.dsl.units`)

Typed wrappers for common musical and physical units, enabling clear and safe conversions:

### `BPM` — Beats Per Minute

```python
from nasong.dsl.units import BPM

tempo = BPM(120)
quarter_ms = tempo.to_ms()          # 500.0 ms
eighth_ms = tempo.to_ms(0.5)        # 250.0 ms
sixteenth_ms = tempo.to_ms(0.25)    # 125.0 ms
```

### `Ms` — Milliseconds

```python
from nasong.dsl.units import Ms

attack = Ms(50)
attack_sec = attack.to_seconds()    # 0.05
```

### `Bars`

```python
from nasong.dsl.units import Bars

loop_length = Bars(4)
# Use with BPM to calculate duration:
# 4 bars at 120 BPM (4/4) = 4 * 4 * (60/120) = 8 seconds
```

### `Hz` — Hertz

```python
from nasong.dsl.units import Hz

frequency = Hz(440.0)
```

---

## Limitations

- The DSL chaining system currently only includes `Gain` as a built-in processor. More processors (filters, delays, reverbs) can be added by subclassing `Processor`.
- The decorators are currently markers only — no automatic registration or validation is performed at runtime.
- The units module provides conversion helpers but does not auto-integrate with `Value` constructors.

---

## Related Documentation

- [Core Values API](api/core_values.md) — All available Value nodes
- [Effects Reference](api/effects.md) — Pre-built effects (ADSR_Piano, Vibrato, etc.)
- [Song Scripting Guide](song_scripting.md) — Standard composition workflow
