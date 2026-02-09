import sounddevice as sd
import numpy as np
import importlib.util
import sys
import threading
import traceback
from typing import Optional, Any
from nasong.core.value import Value
from nasong.app.render_engine import RenderEngine


class LiveSession:
    """
    Manages the audio stream and the 'hot' user code.
    """

    def __init__(self, sample_rate=44100, block_size=2048, device=None):
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

        # Initialize RenderEngine
        self.render_engine = RenderEngine(
            sample_rate=sample_rate, chunk_size=block_size
        )

    def set_error_callback(self, cb):
        self.error_callback = cb

    def set_log_callback(self, cb):
        self.log_callback = cb

    def log(self, msg: str):
        # Use sys.__stdout__ to avoid infinite loops when redirecting globals
        sys.__stdout__.write(str(msg) + "\n")
        sys.__stdout__.flush()
        if hasattr(self, "log_callback") and self.log_callback:
            self.log_callback(msg)

    def set_volume(self, vol: float):
        self.volume = max(0.0, vol)  # ALLOW Over-amplification for testing!

    def load_script(self, script_path: str) -> bool:
        """
        Loads or reloads the user script.
        """
        module_name = "user_script_live"

        # Custom stream to capture prints from user script
        class LogStream:
            def __init__(self, logger):
                self.logger = logger

            def write(self, text):
                if text.strip():
                    self.logger(text.rstrip())

            def flush(self):
                pass

        log_stream = LogStream(self.log)

        try:
            # Check if file exists
            with open(script_path, "r"):
                pass

            spec = importlib.util.spec_from_file_location(module_name, script_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module

                # Capture stdout/stderr during execution
                from contextlib import redirect_stdout, redirect_stderr

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

                        # Create a Time Variable, pass it to song(), and the result is the sequencer Value.

                        from nasong.core.values.basic.value_identity import Identity

                        time_var = Identity()
                        sequencer = module.song(time_var)

                        if not isinstance(sequencer, Value):
                            raise ValueError(
                                "Function 'song' must return a nasong.core.Value"
                            )

                    except Exception as e:
                        self.log(f"Error rendering 'song' function: {e}")
                        sequencer = None

                if sequencer:
                    with self.lock:
                        self.user_module = module
                        self.reload_cursor = self.cursor  # Set marker
                        # Pass sequencer to RenderEngine
                        self.render_engine.set_sequencer(sequencer)
                        self.log(f"Loaded {script_path}")
                    return True
                else:
                    err = f"Error: {script_path} must define 'sequencer' variable or 'song' function returning nasong.core.Value."
                    self.log(err)
                    if self.error_callback:
                        self.error_callback(err)
                    return False
        except Exception as e:
            err = f"Failed to load script: {e}\n{traceback.format_exc()}"
            self.log(err)
            if self.error_callback:
                self.error_callback(err)
            return False
        return False

    def audio_callback(self, outdata, frames, time_info, status):
        if status:
            print(f"Stream status: {status}")

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

                    # We need to fill audio_buffer from current_sample to current_sample + needed_samples

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

                except Exception as e:
                    print(f"Error during audio generation: {e}")
                    outdata.fill(0)
            else:
                if self.log_callback and self.cursor % (self.sample_rate * 5) < frames:
                    self.log_callback("Status: No sequencer loaded.")
                outdata.fill(0)

        self.cursor += frames
        # Update RenderEngine cursor priority
        self.render_engine.update_cursor(self.cursor / self.sample_rate)

    def seek(self, time_seconds: float):
        """
        Seeks to a specific time in seconds.
        """
        with self.lock:
            self.cursor = int(time_seconds * self.sample_rate)
            self.render_engine.update_cursor(time_seconds)
            self.log(f"Seek to {time_seconds:.2f}s")

    def start(self):
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
        except Exception as e:
            print(f"Failed to start audio stream: {e}")
            if self.error_callback:
                self.error_callback(str(e))

    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.render_engine.stop()
