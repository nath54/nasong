# Copyright (C) 2026 Nathan Cerisara <https://github.com/nath54/nasong>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""
Live audio session management for NaSong.

This module provides the `LiveSession` class, which handles the real-time audio
stream, dynamic loading and execution of user scripts, and coordination with
the rendering engine.
"""

#
### Import Modules. ###
#
from typing import Optional, Any, Callable
import importlib.util
from contextlib import redirect_stdout, redirect_stderr

#
import sys
import threading
import traceback
import numpy as np
import sounddevice as sd

#
from nasong.core.value import Value
from nasong.app.render_engine import RenderEngine
from nasong.core.values.basic.value_identity import Identity


class LiveSession:
    """
    Manages a live audio session, including the stream and hot-reloaded user code.

    The LiveSession acts as the bridge between the PortAudio stream (via sounddevice),
    the RenderEngine which computes audio in the background, and the user-provided
    Python scripts that define the music.

    Attributes:
        sample_rate: The audio sample rate (e.g., 44100).
        block_size: Smallest unit of audio processing (buffer size).
        device: PortAudio device index or name.
        cursor: Current absolute sample position in the timeline.
        is_running: Boolean indicating if the audio stream is active.
        volume: Master gain multiplier.
    """

    def __init__(
        self, sample_rate: int = 44100, block_size: int = 2048, device: Any = None
    ):
        """
        Initializes the LiveSession with audio parameters.

        Args:
            sample_rate: Hardware sample rate.
            block_size: Buffer size for each audio callback.
            device: Optional audio device identifier.
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device
        self.stream: Optional[sd.OutputStream] = None
        self.user_module: Optional[Any] = None
        self.cursor = 0  # absolute sample index
        self.is_running = False
        self.lock = threading.Lock()
        self.error_callback: Optional[callable] = None
        self.volume = 0.8
        self.log_callback: Optional[callable] = None
        self.reload_cursor: Optional[int] = None  # Marker for visualization
        self.current_script_path: Optional[str] = None  # For re-render callback

        # Initialize RenderEngine
        self.render_engine = RenderEngine(
            sample_rate=sample_rate, chunk_size=block_size
        )

    def set_error_callback(self, cb: callable):
        """
        Sets a callback function for asynchronous error reporting.

        Args:
            cb: A callable that accepts a string error message.
        """
        self.error_callback = cb

    def set_log_callback(self, cb: callable):
        """
        Sets a callback function for application logging.

        Args:
            cb: A callable that accepts a string log message.
        """
        self.log_callback = cb

    def log(self, msg: str):
        """
        Logs a message to stdout and the registered log callback.

        Args:
            msg: The message to log.
        """
        # Use sys.__stdout__ to avoid infinite loops when redirecting globals
        sys.__stdout__.write(str(msg) + "\n")
        sys.__stdout__.flush()
        if hasattr(self, "log_callback") and self.log_callback:
            self.log_callback(msg)

    def set_volume(self, vol: float):
        """
        Sets the master output volume.

        Args:
            vol: Volume multiplier. Values > 1.0 are allowed for amplification.
        """
        self.volume = max(0.0, vol)  # ALLOW Over-amplification for testing!

    def load_script(self, script_path: str) -> bool:
        """
        Loads or reloads a user script from the filesystem.

        This method dynamically imports the script as a module, captures its output,
        and extracts either a 'sequencer' Variable or a 'song' function to use
        as the audio source.

        Args:
            script_path: Absolute path to the .py user script.

        Returns:
            True if the script was successfully loaded and initialized, False otherwise.
        """
        module_name = "user_script_live"

        # Custom stream to capture prints from user script
        class LogStream:
            """
            Custom stream to capture prints from user script.
            """

            def __init__(self, logger: Callable[[str], None]) -> None:
                """
                Initializes the LogStream with a logger function.

                Args:
                    logger: A callable that accepts a string message.
                """
                self.logger = logger

            def write(self, text: str) -> None:
                """
                Writes text to the logger if it is not empty.

                Args:
                    text: The text to write.
                """
                if text.strip():
                    self.logger(text.rstrip())

            def flush(self) -> None:
                """
                Flushes the stream.
                """
                pass  # pylint: disable=unnecessary-pass

        log_stream = LogStream(self.log)

        try:
            # Check if file exists
            with open(script_path, "r", encoding="utf-8"):
                pass

            spec = importlib.util.spec_from_file_location(module_name, script_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module

                # Capture stdout/stderr during execution
                with redirect_stdout(log_stream), redirect_stderr(log_stream):
                    self.log(f"Executing module: {module_name}")
                    spec.loader.exec_module(module)
                self.log(f"Module execution complete for: {module_name}")

                sequencer = None

                # Check for 'sequencer'
                if hasattr(module, "sequencer") and isinstance(module.sequencer, Value):
                    sequencer = module.sequencer

                # Check for 'song' function
                elif hasattr(module, "song") and callable(module.song):
                    self.log("Found 'song' function. Rendering...")
                    try:
                        # We need to determine BPM? Defaults to 120 if not specified probably
                        # render(song_func, duration=None, bpm=120)
                        # The `song` function in examples usually takes `time` and returns Value.
                        # `render` in nasong.theory expects a `progression` usually?
                        # Let's check how `song_drum_beat.py` is used.
                        # It defines `def song(time: lv.Value) -> lv.Value`.
                        # This is a raw value function, not a Progression.

                        # If it's a raw function `f(time) -> Value`, we can just instantiate it?
                        # But `render` might not be the right tool if it expects a Progression.
                        # If `song` returns a Value, that Value IS the sequencer (conceptually).
                        # But we need to pass a Time Value to it?

                        # Let's look at `song_drum_beat.py`:
                        # def song(time: lv.Value) -> lv.Value:

                        # Create a Time Variable, pass it to song(),
                        # and the result is the sequencer Value.

                        time_var = Identity()
                        sequencer = module.song(time_var)

                        if not isinstance(sequencer, Value):
                            raise ValueError(
                                "Function 'song' must return a nasong.core.Value"
                            )

                    except Exception as e:  # pylint: disable=broad-except
                        self.log(f"Error rendering 'song' function: {e}")
                        sequencer = None

                if sequencer:  # pylint: disable=no-else-return
                    with self.lock:
                        self.user_module = module
                        self.reload_cursor = self.cursor  # Set marker
                        self.current_script_path = script_path
                        # Pass sequencer to RenderEngine
                        self.render_engine.set_sequencer(sequencer)

                        # Check for FORCE_RERENDER_EVERY global parameter
                        force_rerender = getattr(module, "FORCE_RERENDER_EVERY", 0)
                        if force_rerender and isinstance(force_rerender, (int, float)):
                            self.render_engine.set_force_rerender_every(
                                int(force_rerender)
                            )
                            self.render_engine.set_rerender_callback(
                                self._on_force_rerender
                            )
                            self.log(
                                f"Force re-render enabled every"
                                f" {int(force_rerender)} chunks"
                            )
                        else:
                            self.render_engine.set_force_rerender_every(0)
                            self.render_engine.set_rerender_callback(None)

                        self.log(f"Loaded {script_path}")
                    return True
                else:
                    err = f"Error: {script_path} must define 'sequencer' variable or 'song' function returning nasong.core.Value."
                    self.log(err)
                    if self.error_callback:
                        self.error_callback(err)
                    return False
        except Exception as e:  # pylint: disable=broad-except
            err = f"Failed to load script: {e}\n{traceback.format_exc()}"
            self.log(err)
            if self.error_callback:
                self.error_callback(err)
            return False
        return False

    def audio_callback(
        self, outdata: np.ndarray, frames: int, _time_info: Any, _status: Any
    ):
        """
        PortAudio callback for generating the next block of audio.

        This method pulls samples from the RenderEngine, applies master volume,
        and copies them to the output buffer. It also updates the session cursor.

        Args:
            outdata: The output buffer to fill.
            frames: The number of frames requested.
            time_info: Time-related metadata from PortAudio.
            status: Stream status flags.
        """

        # Align with nasong's preference for float32
        # TUI/Session uses cursor to track absolute time
        with self.lock:
            # Periodic logging to help user see if audio is active
            if self.log_callback and self.cursor % (self.sample_rate * 2) < frames:
                # We calculate peak before scaling for the log
                # This peak is calculated AFTER the engine filling it?
                # Actually outdata is filled by us. Let's calculate peak of the generated audio.
                pass

            if self.render_engine.sequencer:
                # seq = self.render_engine.sequencer # Accessible via engine

                try:
                    # Request audio from RenderEngine
                    # We might need to handle buffer size mismatches if engine chunk != frames
                    # Ideally frames == block_size == engine.chunk_size

                    # For now, let's assume frames matched.
                    # But PortAudio might vary frames.

                    # Construct buffer from potentially multiple chunks or partial chunks?
                    # RenderEngine is chunk-based.

                    # Simplest approach: RenderEngine just computes the exact needed range?
                    # No, we want caching of fixed blocks.

                    # Let's just ask RenderEngine for samples frame by frame (or small blocks)
                    # But RenderEngine.get_audio_chunk returns a full chunk.

                    # Re-design RenderEngine get_audio_chunk to return specific range?
                    # Actually, let's do simple logic here:

                    current_sample = self.cursor
                    needed_samples = frames

                    audio_buffer = np.zeros(frames, dtype=np.float32)

                    # We need to fill audio_buffer from current_sample
                    # to current_sample + needed_samples

                    # Iterate through needed range
                    filled = 0
                    while filled < needed_samples:
                        target = current_sample + filled

                        # Find which chunk this target belongs to
                        chunk_start = (
                            target // self.render_engine.chunk_size
                        ) * self.render_engine.chunk_size
                        chunk_offset = target - chunk_start

                        chunk, _ = self.render_engine.get_audio_chunk(chunk_start)

                        if chunk is None:
                            # Cache miss or not ready.
                            # We can output silence or try to render immediately (blocking)?
                            # Provide silence for now to avoid stuttering/blocking audio thread
                            # Or maybe a lightweight heuristic?
                            # For now: Silence (or extrapolation?)
                            # Let's render synchronously if missing? No, that defeats the purpose.
                            # Just Silence.
                            break

                        # How much can we copy from this chunk?
                        available = len(chunk) - chunk_offset
                        to_copy = min(needed_samples - filled, available)

                        audio_buffer[filled : filled + to_copy] = chunk[
                            chunk_offset : chunk_offset + to_copy
                        ]
                        filled += to_copy

                    audio = audio_buffer

                    # Apply Master Volume
                    audio = audio * self.volume

                    # Safety Clip
                    audio = np.clip(audio, -1.0, 1.0)

                    if (
                        self.log_callback
                        and self.cursor % (self.sample_rate * 2) < frames
                    ):
                        peak = float(np.max(np.abs(audio)))
                        self.log_callback(
                            f"Audio peak: {peak:.4f} (Vol: {self.volume:.1f})"
                        )

                    # Stereo Duplication
                    outdata[:, 0] = audio
                    outdata[:, 1] = audio

                except Exception as e:  # pylint: disable=broad-except
                    print(f"Error during audio generation: {e}")
                    outdata.fill(0)
            else:
                if self.log_callback and self.cursor % (self.sample_rate * 5) < frames:
                    self.log_callback("Status: No sequencer loaded.")
                outdata.fill(0)

        self.cursor += frames
        # Update RenderEngine cursor priority
        self.render_engine.update_cursor(self.cursor / self.sample_rate)

    def _on_force_rerender(self):
        """
        Callback invoked by RenderEngine when a forced re-render is triggered.

        Re-executes the current user script to regenerate random values,
        producing a fresh sequencer for subsequent chunks.
        """
        if self.current_script_path:
            self.log("Force re-render: re-executing script for fresh random values...")
            self.load_script(self.current_script_path)

    def seek(self, time_seconds: float):
        """
        Seeks the session to a specific point in time.

        Args:
            time_seconds: The destination time in seconds.
        """
        with self.lock:
            self.cursor = int(time_seconds * self.sample_rate)
            self.render_engine.update_cursor(time_seconds)
            self.log(f"Seek to {time_seconds:.2f}s")

    def start(self):
        """
        Starts the real-time audio output stream.
        """
        if self.is_running:
            return
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=self.device,
                channels=2,
                callback=self.audio_callback,
            )
            self.stream.start()
            self.is_running = True
        except Exception as e:  # pylint: disable=broad-except
            print(f"Failed to start audio stream: {e}")
            if self.error_callback:
                self.error_callback(str(e))

    def stop(self):
        """
        Stops the audio stream and shuts down the rendering engine.
        """
        if not self.is_running:
            return
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.render_engine.stop()
