import sounddevice as sd
import numpy as np
import importlib.util
import sys
import threading
import traceback
from typing import Optional, Any
from nasong.core.value import Value


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

                # Check for 'sequencer'
                if hasattr(module, "sequencer") and isinstance(module.sequencer, Value):
                    with self.lock:
                        self.user_module = module
                        self.reload_cursor = self.cursor  # Set marker
                        self.log(f"Loaded {script_path}")
                    return True
                else:
                    err = f"Error: {script_path} must define 'sequencer' variable of type nasong.core.Value."
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

            if self.user_module and hasattr(self.user_module, "sequencer"):
                seq = self.user_module.sequencer

                # Generate time array for this block (seconds)
                t_start = self.cursor / self.sample_rate
                t_end = (self.cursor + frames) / self.sample_rate
                time_array = np.linspace(
                    t_start, t_end, frames, endpoint=False, dtype=np.float32
                )

                try:
                    # Get audio from sequencer
                    audio = seq.getitem_np(time_array, self.sample_rate)

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
