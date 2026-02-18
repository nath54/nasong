# Style Generators API

Per-style documentation for all generator classes in `nasong.theory.generators.styles`.

Each generator provides static methods that return `Progression`, `Rhythm`, or both.

---

## Jazz

```python
from nasong.theory.generators.styles.jazz import Jazz
```

Generates sophisticated Jazz structures: standard turnarounds and random lead-sheet progressions.

### `Jazz.ii_V_I(root="C4", minor=False) → Progression`

Generates a classic ii-V-I turnaround.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `root` | `str` | `"C4"` | The tonic note |
| `minor` | `bool` | `False` | Use minor ii-V-i form |

**Returns:** `Progression` with 3 chords.

**Example:**
```python
prog = Jazz.ii_V_I("C4")           # Major: Dm - G - C
prog = Jazz.ii_V_I("A4", minor=True)  # Minor: Bm7b5 - E7 - Am
```

### `Jazz.generate_random_standards_progression(length=4) → Progression`

Builds a random progression from a pool of idiomatic jazz patterns (ii-V-I, rhythm changes, backdoor cadence, etc.).

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `length` | `int` | `4` | Desired number of chords |

**Returns:** `Progression` with `length` chords in a random key.

**Example:**
```python
prog = Jazz.generate_random_standards_progression(8)
```

---

## EDM

```python
from nasong.theory.generators.styles.edm import EDM
```

High-energy Electronic Dance Music components.

### `EDM.epic_chords(root="F4") → Progression`

Generates an "epic" dance chord progression (vi-IV-I-V).

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `root` | `str` | `"F4"` | The root key |

**Returns:** `Progression` with 4 chords.

**Example:**
```python
prog = EDM.epic_chords("F4")  # Am - F - C - G (in key of C)
```

### `EDM.basic_beat() → Rhythm`

Returns a standard "four on the floor" kick rhythm pattern.

**Returns:** `Rhythm`

---

## Lofi

```python
from nasong.theory.generators.styles.lofi import Lofi
```

Relaxed, atmospheric Lofi Hip Hop progressions.

### `Lofi.chill_progression(root="Db4") → Progression`

Generates a relaxed ii-V-I style progression with extended jazz voicings.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `root` | `str` | `"Db4"` | The root key |

**Returns:** `Progression` with 3 chords.

**Example:**
```python
prog = Lofi.chill_progression("Eb4")
```

---

## Salsa

```python
from nasong.theory.generators.styles.salsa import Salsa
```

Afro-Cuban harmonic montunos and clave patterns.

### `Salsa.montuno_progression(root="G4", minor=True) → Progression`

Generates a basic i-V salsa montuno vamp.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `root` | `str` | `"G4"` | Root note |
| `minor` | `bool` | `True` | Use minor key |

**Returns:** `Progression` with 2 chords.

### `Salsa.clave_rhythm(direction="2-3") → Rhythm`

Returns the iconic Son Clave pattern as a 16-step (sixteenth-note) grid.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `direction` | `str` | `"2-3"` | `"2-3"` or `"3-2"` clave direction |

**Returns:** `Rhythm` with 16 sixteenth-note pulses.

**Raises:** `ValueError` if direction is not `"2-3"` or `"3-2"`.

**Example:**
```python
clave_23 = Salsa.clave_rhythm("2-3")
clave_32 = Salsa.clave_rhythm("3-2")
```

---

## Afrobeat

```python
from nasong.theory.generators.styles.afrobeat import Afrobeat
```

West African-inspired patterns with polyrhythmic structures.

### `Afrobeat.polyrhythmic_groove(root="C4") → Progression`

Generates a basic Afrobeat groove with a 3:2 polyrhythm over an I-IV vamp.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `root` | `str` | `"C4"` | Root note for the pentatonic scale |

**Returns:** `Progression` with additional attributes:
- `prog.rhythm_a` — `Rhythm` for the 3-beat polyrhythm line
- `prog.rhythm_b` — `Rhythm` for the 2-beat polyrhythm line

**Example:**
```python
prog = Afrobeat.polyrhythmic_groove("C4")
print(prog.rhythm_a)  # 3-against-2 pattern
print(prog.rhythm_b)  # 2-against-3 pattern
```

---

## Celtic

```python
from nasong.theory.generators.styles.celtic import Celtic
```

Traditional Celtic and Irish folk patterns.

---

## Bossa Nova

```python
from nasong.theory.generators.styles.bossa_nova import BossaNova
```

Brazilian Bossa Nova style progressions.

---

## Koto

```python
from nasong.theory.generators.styles.koto import Koto
```

Japanese Koto-inspired melodic patterns.

---

## Using Generators with `render()`

All generators produce theory objects (`Progression`, `Rhythm`) that can be fed into `render()` to produce audio:

```python
from nasong.theory import render
from nasong.theory.generators.styles.jazz import Jazz
from nasong.core.all_values import Sin, ADSR2, Constant

# 1. Define an instrument
def my_synth(time, freq, start, dur, vel):
    osc = Sin(time, Constant(freq * 6.2832))
    env = ADSR2(time, start, dur, 0.01, 0.1, 0.6, 0.2)
    return osc * env * vel * 0.4

# 2. Generate a progression
prog = Jazz.ii_V_I("C4")

# 3. Render to audio
sequencer = render(prog, time_value=None, instrument_factory=my_synth, bpm=120)
```

---

## Summary Table

| Style | Class | Key Methods | Output |
| :--- | :--- | :--- | :--- |
| Jazz | `Jazz` | `ii_V_I()`, `generate_random_standards_progression()` | `Progression` |
| EDM | `EDM` | `epic_chords()`, `basic_beat()` | `Progression`, `Rhythm` |
| Lofi | `Lofi` | `chill_progression()` | `Progression` |
| Salsa | `Salsa` | `montuno_progression()`, `clave_rhythm()` | `Progression`, `Rhythm` |
| Afrobeat | `Afrobeat` | `polyrhythmic_groove()` | `Progression` + rhythms |
| Celtic | `Celtic` | — | — |
| Bossa Nova | `BossaNova` | — | — |
| Koto | `Koto` | — | — |

---

## Related Documentation

- [Music Theory Module](../music_theory.md) — Core theory concepts and structures
- [Core Values API](core_values.md) — All Value nodes
- [Song Scripting Guide](../song_scripting.md) — Building complete songs
