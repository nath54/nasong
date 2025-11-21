# Nasong

Nasong is a Python-based music synthesizer and sequencer that allows you to create music programmatically. It provides a framework for defining instruments, effects, and songs using Python code, which are then rendered to WAV files.

## Features

- **Programmatic Music Generation**: Define songs and instruments using Python code.
- **Custom Instruments**: Create your own instruments by defining their waveforms and envelopes.
- **Built-in Library**: Includes a library of basic instruments (strings, winds, percussion, synths) and effects.
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

- **Not Real-Time**: Nasong is a "music compiler". You write code, run the script to render a WAV file, and then listen. It is not designed for live performance or real-time jamming.
- **Requires Coding**: You need to be comfortable with Python to use it effectively.
- **Render Time**: Complex songs with many voices and heavy processing (like convolution reverb) may take some time to render.


## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/nasong/nasong.git
    cd nasong
    ```
2.  Ensure you have Python 3 installed.
3.  (Optional) Install VLC or another media player to play the generated WAV files from the command line.

## Usage

To compile a song and play it, use the `main.py` script. You need to specify the sample rate and the input song file.

### Basic Command

```bash
python main.py -s 44100 -i song_examples/song_electronic_synth.py -o song_electronic_synth.wav && vlc song_electronic_synth.wav
```

### Arguments

- `-i`: Path to the Python song description file (Required).
- `-s`: Sample rate in Hz (Default: 44100).
- `-o`: Path to the generated output WAV file (Default: `output.wav`).

### Example

To render the "Electronic Synth" example:

```bash
python main.py -s 44100 -i song_examples/song_electronic_synth.py -o song_electronic_synth.wav
```

This will create an `song_electronic_synth.wav` file in the current directory.

## Project Structure

- `main.py`: The entry point for the application.
- `lib_*.py`: Core libraries for value generation, song structure, import handling, etc.
- `lib_ext_*.py`: Extended libraries containing instrument definitions (bass, bowed strings, keyboards, plucked strings, winds, percussion, synths).
- `song_examples/`: Directory containing example song definitions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
