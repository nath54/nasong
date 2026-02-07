# Nasong

Nasong is a Python-based music synthesizer and sequencer that allows you to create music programmatically. It provides a framework for defining instruments, effects, and songs using Python code, which are then rendered to WAV files.

## Features

- **Programmatic Music Generation**: Define songs and instruments using Python code.
- **Custom Instruments**: Create your own instruments by defining their waveforms and envelopes.
- **Built-in Library**: Includes a library of basic instruments (strings, winds, percussion, synths) and effects.
- **Trainable Instruments**: Differentiable instruments that can learn parameters from target audio samples.
- **Experiment Tracking**: Built-in system to track training runs, metrics, and parameters.
- **High Quality Output**: Generates standard WAV files.

## Philosophy & Core Concepts

Nasong is built on the philosophy of **"Code as Music"**. Instead of using a graphical DAW (Digital Audio Workstation) with fixed tracks and plugins, you define your music using composable Python objects. This approach treats sound synthesis, composition, and arrangement as a unified programming task.

### The `Value` Class

At the heart of Nasong is the `Value` class.

- **Everything is a Value**: A `Value` represents a signal that varies over time. This could be an audio waveform (like a sine wave), a control signal (like an LFO or envelope), or even a constant number.
- **Composition**: You build complex sounds by combining `Value` objects. For example, a synthesizer might be a `Sin` oscillator whose frequency is modulated by another `Sin` (LFO) and whose amplitude is controlled by an `ADSR` envelope. All of these are `Value` objects.
- **Vectorized Processing**: Under the hood, `Value` objects use NumPy for fast, vectorized processing (`getitem_np`), allowing for efficient rendering of complex audio graphs.

## Benefits

- **Infinite Customization**: You are not limited by the architecture of a specific VST or synthesizer. You can build your own synthesis architectures from scratch.
- **Version Control for Music**: Since your music is plain text code, you can use Git to track changes, branch ideas, and collaborate.
- **Procedural Generation**: Use Python's loops, logic, and random libraries to create generative music, evolving soundscapes, and algorithmic compositions.
- **Precision**: Define exact frequencies, timings, and modulation curves mathematically.

## Constraints

- **Not Real-Time**: Nasong is a "music compiler". You write code, runs the script to render a WAV file, and then listen. It is not designed for live performance or real-time jamming.
- **Requires Coding**: You need to be comfortable with Python to use it effectively.
- **Render Time**: Complex songs with many voices and heavy processing (like convolution reverb) may take some time to render.


## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/nasong/nasong.git
    cd nasong
    ```
2.  Install the package (editable mode recommended for development):
    ```bash
    pip install -e .
    ```
    *Note: PyTorch is an optional dependency for GPU acceleration and training. If you want to use it, install it separately following instructions at [pytorch.org](https://pytorch.org).*

This installation exposes the following CLI commands:
- `nasong`: Generate music.
- `nasong-vis`: Visualize audio.
- `nasong-train`: Train instruments.
- `nasong-monitor`: Manage experiments.

## Usage

### 1. Creating a Nasong File

A "Nasong file" is simply a Python script (e.g., `my_song.py`) that exports the logic for your music.

**Required Structure:**
Your script **MUST** define two things:
1.  **`duration`**: A variable (float/int) specifying the total length in seconds.
2.  **`song(time)`**: A function that takes a `time` Value and returns the final audio output `Value`.

**Example Template:**
```python
import nasong.core.value as lv
from nasong.instruments.synth import SimpleSynth

# 1. Define Duration
duration = 10.0  # seconds

# 2. Define Song Function
def song(time: lv.Value) -> lv.Value:
    # Build your audio graph here
    # 'time' is the global time ramp signal provided by the renderer
    
    # Example: A simple 440Hz sine wave
    intro = SimpleSynth(time, frequency=lv.Constant(440))
    
    return intro
```

### 2. Generating Music (`nasong`)

Use the `nasong` command to compile your song into audio.

```bash
nasong my_song.py -o output.wav
```

**Arguments:**
- `input_file`: Path to the Python song description file.
- `-o`, `--output`: Output WAV filename (Default: `output.wav`).
- `-s`, `--sample-rate`: Sample rate in Hz (Default: 44100).
- `-t`, `--torch`: Use PyTorch for rendering (requires Torch installed).
- `-d`, `--device`: Device to use (e.g., `cpu`, `cuda`).

### 3. Visualizing Audio (`nasong-vis`)

Analyze or plot waveforms/spectrograms of generated audio.

```bash
nasong-vis -i output.wav --analyze --plot spectrogram
```

### 4. Training Instruments (`nasong-train`)

You can train generative instruments to match a target audio sample (e.g., make a synth sound like a specific recording).

```bash
nasong-train --instrument named_fm --target my_sample.wav --epochs 1000
```

This will:
- Run an optimization loop using PyTorch.
- Log metrics (loss, duration) to `~/.nasong/experiments/`.
- Save the learned parameters to `params.json`.

### 5. Monitoring Experiments (`nasong-monitor`)

Manage your training experiments.

- **List experiments**:
  ```bash
  nasong-monitor list
  ```
- **Show details**:
  ```bash
  nasong-monitor show <experiment_id>
  ```
- **Delete experiment**:
  ```bash
  nasong-monitor delete <experiment_id>
  ```

### 6. Evaluating Models (`nasong-evaluate`)

Evaluate the performance of trained models by comparing detected notes in the target vs. synthesized audio.

```bash
nasong-evaluate --experiment my_experiment
```
Or evaluate all experiments in a directory:
```bash
nasong-evaluate --models-dir trained_models
```

This generates:
- `evaluation.json`: Detailed note detection metrics.
- `comparison_<instrument>.png`: Side-by-side spectrograms.

### 7. Leaderboards (`nasong-leaderboard`)

Generate a global leaderboard comparing all trained models.

```bash
nasong-leaderboard --output results_analysis/leaderboards.md
```

## Experiment Tracking & Inference

Nasong allows you to use trained instruments in your songs **without** needing PyTorch installed. The system effectively "compiles" the trained parameters into the instrument.

### Using Trained Instruments

Use the `load_trained_instrument` helper to load an instrument with its trained parameters pre-injected.

```python
from nasong.trainable.inference import load_trained_instrument
import nasong.core.value as lv

# 1. Load instrument from Experiment ID (get this from nasong-monitor list)
# This returns a callable function identical to the original blueprint but with defaults updated.
my_instrument = load_trained_instrument("a1b2c3d4") 

# 2. Use it in your song graph
# It behaves exactly like a normal instrument
def song(time: lv.Value) -> lv.Value:
    return my_instrument(
        time=time, 
        frequency=lv.Constant(440), 
        start_time=0.0, 
        duration=1.0
    )
```

## Project Structure

- `src/nasong/core/`: Core libraries (Values, Song, Wav, config).
- `src/nasong/instruments/`: Built-in instrument library.
- `src/nasong/scripts/`: CLI entry points.
- `src/nasong/trainable/`: Training logic and trainable instrument definitions.
- `song_examples/`: Example song definitions.
- `tests/`: Automated tests.
