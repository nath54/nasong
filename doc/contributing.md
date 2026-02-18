# Contributing to NaSong

Thank you for your interest in contributing to NaSong! This guide covers the development setup, coding standards, and step-by-step checklists for common contribution types.

---

## Development Setup

### 1. Fork & Clone

```bash
git clone https://github.com/<your-username>/nasong.git
cd nasong
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

### 3. Install in Editable Mode

```bash
pip install -e ".[dev]"
```

Or install the base package and dev dependencies separately:

```bash
pip install -e .
pip install -r requirements.txt
```

### 4. Verify the Installation

```bash
nasong --help
nasong-rave --help
```

---

## Code Style

### Docstrings

Use **Google-style** docstrings for all public classes, methods, and functions:

```python
def my_function(param1: int, param2: str = "default") -> bool:
    """Brief one-line summary.

    Extended description if needed.

    Args:
        param1 (int): Description of param1.
        param2 (str, optional): Description of param2. Defaults to "default".

    Returns:
        bool: Description of return value.

    Raises:
        ValueError: If param1 is negative.
    """
```

### Type Hints

All function signatures should include type hints:

```python
def render_chunk(
    self,
    start_sample: int,
    chunk_size: int,
    sample_rate: int,
) -> NDArray[np.float32]:
```

### Linting

We use **Pylint** and **Mypy** for static analysis:

```bash
pylint src/nasong/
mypy src/nasong/
```

Key conventions:
- No wildcard imports (`from x import *`)
- Constants in `UPPER_SNAKE_CASE`
- Classes in `PascalCase`
- Functions and variables in `snake_case`
- Keep functions under ~50 lines where possible

---

## Running Tests

### Test Suite

```bash
pytest
```

### Test Generation

A test generation script is available to auto-generate pytest stub files for new modules:

```bash
python scripts/generate_tests.py
```

This scans `src/nasong/` and creates corresponding test files in `tests/` without overwriting existing test code.

---

## Adding a New Instrument

### Checklist

1. **Choose the right file** in `src/nasong/instruments/`:
   - `bass.py` — Bass instruments
   - `keyboards.py` — Piano, organ, etc.
   - `synth.py` — Synthesizer leads and pads
   - `percussion.py` — Drums and percussion
   - `bowed_strings.py` — Violin, cello, etc.
   - `plucked_strings.py` — Guitar, harp, etc.
   - `winds.py` — Flute, trumpet, etc.

2. **Define the instrument factory function**:
   ```python
   def MyNewInstrument(
       time: Value,
       freq: float,
       start: float,
       duration: float,
       velocity: float,
       sample_rate: int,
   ) -> Value:
       """Descriptive docstring.

       Args:
           time: Global time Value.
           freq: Note frequency in Hz.
           start: Note start time in seconds.
           duration: Note duration in seconds.
           velocity: Note velocity (0.0–1.0).
           sample_rate: Audio sample rate.

       Returns:
           Value: The audio signal for this note.
       """
       osc = Sin(time, Constant(freq * 6.2832))
       env = ADSR2(time, start, duration, 0.01, 0.1, 0.6, 0.2)
       return osc * env * velocity
   ```

3. **Add a Google-style docstring** with parameter descriptions.

4. **Test it** with a simple script:
   ```python
   from nasong.theory import render
   from nasong.theory.systems.western import Western
   from nasong.theory.structures.progression import Progression
   from nasong.instruments.your_file import MyNewInstrument

   prog = Progression.from_roman_numerals(Western.major("C4"), ["I", "V"])
   seq = render(prog, None, MyNewInstrument, bpm=120)
   ```

5. **Document it** in `doc/instruments.md`.

---

## Adding a New Value Node

### Checklist

1. **Create the file**: `src/nasong/core/values/<category>/value_my_node.py`

2. **Implement the class** — Must subclass `Value` and implement:
   - `__init__` — Store child values and parameters
   - `get_item(index, sample_rate) → float` — Scalar fallback
   - `getitem_np(indexes_buffer, sample_rate) → NDArray` — NumPy vectorized

3. **Optional implementations**:
   - `getitem_torch(indexes_buffer, sample_rate, device) → Tensor` — For GPU/autograd
   - `backward(grad_output, context, sample_rate)` — For custom gradients

4. **Register** in `src/nasong/core/all_values.py`:
   ```python
   from nasong.core.values.<category>.value_my_node import MyNode
   # Add "MyNode" to __all__
   ```

5. **Write tests** in `tests/`.

6. **Document** in `doc/api/core_values.md`.

See [Value System](value_system.md) for a detailed walkthrough.

---

## Adding a New Music Theory System

### Checklist

1. **Create the file**: `src/nasong/theory/systems/my_system.py`

2. **Define the system class** with scale factory methods:
   ```python
   from nasong.theory.core.scale import Scale

   class MySystem:
       """Description of the musical tradition."""

       @staticmethod
       def pentatonic(root: str) -> Scale:
           """Returns the pentatonic scale."""
           return Scale.from_intervals(root, [0, 2, 4, 7, 9])
   ```

3. **Add tests** in `tests/`.

4. **Document** in `doc/music_theory.md` under the Scale Systems section.

---

## Adding a New Style Generator

### Checklist

1. **Create the file**: `src/nasong/theory/generators/styles/my_style.py`

2. **Define the generator class** with static methods:
   ```python
   from nasong.theory.structures.progression import Progression

   class MyStyle:
       """Description of the genre."""

       @staticmethod
       def signature_progression(root: str = "C4") -> Progression:
           """Returns a characteristic chord progression."""
           ...
   ```

3. **Add tests** in `tests/`.

4. **Document** in `doc/api/theory_generators.md`.

---

## PR Guidelines

1. **Branch naming**: `feature/my-feature`, `fix/bug-description`, or `docs/update-name`
2. **Commit messages**: Use imperative mood (e.g., "Add new violin instrument")
3. **Tests**: Include tests for any new functionality
4. **Docstrings**: All public APIs must have Google-style docstrings
5. **Type hints**: All function signatures must have type annotations
6. **Lint clean**: Run `pylint` and `mypy` before submitting
7. **Description**: Explain *what* and *why* in your PR description

---

## Related Documentation

- [Architecture](architecture.md) — Project-wide module overview
- [Value System](value_system.md) — Deep dive into the Value engine
- [Instruments](instruments.md) — Existing instrument reference
