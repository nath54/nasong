# Live Coding Paradigm Options

The goal is to move away from the low-level `def song(t_vector)` approach and find a "User Code" syntax that feels musical, intuitive, and robust for live performance.

Here are 3 distinct options for how the "Chunk System" can present itself to you.

---

## Option 1: The "Sonic Pi" Style (Imperative Coroutines)
**Paradigm**: You write loops that `sleep` or `wait`.
**Pros**: Extremely intuitive for rhythm and structure. You read it like a story.
**Cons**: Hardest to implement (requires a scheduler that "looks ahead" to fill audio chunks).

### User Code Example:
```python
@session.loop(name="drums")
def drums():
    # Loop runs forever until code changes
    play(Kick)
    wait(1/4) # Wait quarter beat
    play(Snare)
    wait(1/4)

@session.loop(name="bass")
def bass():
    # Syncs automatically to next bar
    progression = ["C2", "F2", "G2", "C2"]
    
    for note in progression:
        # Trigger synth with parameters
        start(BassSynth, freq=note, decay=0.2)
        wait(1) # Wait 1 bar
```

**How it works with Chunks:**
The engine runs these functions in a "virtual time" thread. It collects all the `play()` events scheduled for the next 100ms (the chunk), renders them, and mixes them. The user never sees `t`.

---

## Option 2: The "TidalCycles" Style (Pattern Declarations)
**Paradigm**: You define *infinite patterns* of events.
**Pros**: Very concise. Great for complex polyrhythms. Purely functional/declarative.
**Cons**: Can feel abstract. Less "programming", more "equation solving".

### User Code Example:
```python
# You define the "What" and "When" as data structures
# The engine automatically slices this data for the current audio chunk

d1 = Track(
    sound=Kick,
    # "Play kick 4 times per cycle"
    pattern="x x x x"
)

d2 = Track(
    sound=Snare,
    # "Play snare on offbeats, with euclidean variation"
    pattern="~ x ~ x"
)

bass = Track(
    sound=FM_Bass,
    # Apply melody pattern to synth
    notes="<C2 F2 G2 C2>", # Cycles every bar
    # Apply separate rhythm
    struct="x(3,8)" # 3 hits in 8 steps (Euclidean)
)

# You just update these variable definitions live.
play(d1, d2, bass)
```

**How it works with Chunks:**
Every audio chunk (e.g. 50ms), the engine calculates: "Where am I in the cycle?" (e.g. 0.25 to 0.30). It queries the pattern strings: "Are there events in this window?". If yes, it renders them.

---

## Option 3: The "Ableton Clip" Style (Object-Oriented Session)
**Paradigm**: You have a persistent `Session` object. You sequence "Clips" (functions/generators) into tracks.
**Pros**: Familiar to DAW users. Good balance of control and automation.
**Cons**: Slightly more verbose than Pattern style.

### User Code Example:
```python
# 1. Define your Instruments (Sound design)
kick = Kick909(tune=50)
bass = MoogBass(cutoff=500)

# 2. Define Patterns (Clips)
def techno_beat(ctx):
    # imperatively add events to the current grid
    ctx.kick(0, 1, 2, 3) 
    ctx.hats(0.5, 1.5, 2.5, 3.5)

def rolling_bass(ctx):
    ctx.note(bass, "C2", start=0, dur=0.25)
    ctx.note(bass, "C2", start=0.75, dur=0.25)

# 3. The "Live" part: Launching Clips
# You edit this part live to switch sections
session.bpm = 128
session.track("Drums").play(techno_beat)
session.track("Bass").play(rolling_bass) 
```

**How it works with Chunks:**
The Session holds the state. Every chunk, it advances its internal transport. It calls the active clips to ask "What notes are playing right now?".

---

## Technical Comparison

| Feature | Option 1 (Imperative) | Option 2 (Patterns) | Option 3 (Session/Clips) |
| :--- | :--- | :--- | :--- |
| **Logic** | `wait()` / `sleep()` | `"x x x x"` strings | `play(clip_A)` |
| **State** | Implicit (program counter) | Stateless (Time functions) | Object State (Session) |
| **Complexity** | Low (easy to read) | High (dense syntax) | Medium (Pythonic) |
| **Flexibility**| Best for linear structures | Best for wild rhythms | Best for arrangements |

Which direction resonates more with your workflow?
