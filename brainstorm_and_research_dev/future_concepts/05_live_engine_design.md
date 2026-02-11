# 05_live_engine_design.py

```py
# TECHNICAL PROTOTYPE: NaSong Live Engine Internals
# This file demonstrates HOW the "Chunk System" works under the hood.

import time
import importlib
import numpy as np
import threading
from typing import Callable, Any


# Mocking Audio I/O
class AudioStream:
    def write(self, data: np.ndarray):
        print(f"[Audio Hardware] Playing chunk: {len(data)} samples")


# ==========================================
# 1. State Container
# ==========================================
# To survive reloads, data must live in an external object passed to the user code.


class SessionState(dict):
    """
    Persistent state dictionary that survives module reloads.
    Users can store counters, pattern indices, or random seeds here.
    """

    def get_or_set(self, key, default):
        if key not in self:
            self[key] = default
        return self[key]


# ==========================================
# 2. The Live Engine
# ==========================================


class LiveEngine:
    def __init__(self, user_script_path: str, sample_rate=44100, chunk_size=2048):
        self.user_script_path = user_script_path
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stream = AudioStream()

        # Timing
        self.global_time = 0.0  # Seconds
        self.frame_index = 0  # Samples

        # The User's Code
        self.user_module = None
        self.current_song_func = None

        # Hot-Swap Logic
        self.next_song_func = None  # Buffered for crossfade
        self.is_fading = False
        self.fade_progress = 0.0
        self.CROSSFADE_DURATION = 0.05  # 50ms

        # Persistence
        self.state = SessionState()

        # Threading
        self.running = False

    # ------------------------------------------
    # A. Reload Mechanism
    # ------------------------------------------
    def reload_code(self):
        """
        Called by file watcher when script changes.
        """
        try:
            print(f"Reloading {self.user_script_path}...")

            # 1. Import or Reload the python module
            if self.user_module:
                self.user_module = importlib.reload(self.user_module)
            else:
                self.user_module = importlib.import_module(self.user_script_path)

            # 2. Extract the 'song' function
            # The standard protocol is: def song(t, state): return freq_value
            new_func = getattr(self.user_module, "song", None)

            if not new_func:
                print("Error: No 'song' function found in module.")
                return

            # 3. Schedule the Swap
            if self.current_song_func is None:
                self.current_song_func = new_func
            else:
                self.next_song_func = new_func
                self.is_fading = True
                self.fade_progress = 0.0
                print("Hot-swap scheduled with crossfade.")

        except SyntaxError as e:
            print(f"Syntax Error in user code: {e}")
            # Do NOT stop playback, keep running old code
        except Exception as e:
            print(f"Runtime Error during reload: {e}")

    # ------------------------------------------
    # B. The Audio Loop (Chunk Generator)
    # ------------------------------------------
    def run_loop(self):
        self.running = True

        while self.running:
            start_time = time.time()

            # 1. Determine Time Range for this Chunk
            t_start = self.frame_index / self.sample_rate
            t_end = (self.frame_index + self.chunk_size) / self.sample_rate

            # Create a Time Value for this block (vectorized)
            # This is passed to the NaSong graph
            # t_tensor shape: [chunk_size]
            t_tensor = np.linspace(t_start, t_end, self.chunk_size, endpoint=False)

            # 2. Render Audio
            audio_chunk = self._render_next_chunk(t_tensor)

            # 3. Output to stream (blocking or buffering)
            self.stream.write(audio_chunk)

            # 4. Advance Clock
            self.frame_index += self.chunk_size
            self.global_time = t_end

            # Sleep to emulate real-time (not needed if stream.write is blocking)
            elapsed = time.time() - start_time
            sleep_time = (self.chunk_size / self.sample_rate) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ------------------------------------------
    # C. Rendering & Crossfading
    # ------------------------------------------
    def _render_next_chunk(self, t_tensor):
        """
        The Core Logic: Handling the Graph generation and Hot-Swap.
        """

        # Edge Case: No code loaded yet
        if not self.current_song_func:
            return np.zeros_like(t_tensor)

        # 1. Build the Graph for the current function
        # We pass 't_tensor' (Time) and 'self.state' (Persistence)
        # In NaSong, 'song(t)' returns a Value object (the graph)
        graph_a = self.current_song_func(t_tensor, self.state)

        # Render the graph to raw audio samples
        # audio_a = graph_a.render_np(t_tensor)
        # (Mocking return)
        audio_a = np.sin(2 * np.pi * 440 * t_tensor)  # Placeholder sine

        # 2. Handle Crossfade if Swapping
        if self.is_fading and self.next_song_func:
            # Build Graph B
            graph_b = self.next_song_func(t_tensor, self.state)

            # audio_b = graph_b.render_np(t_tensor)
            audio_b = np.sin(2 * np.pi * 880 * t_tensor)  # Placeholder

            # Calculate Crossfade envelops for this chunk
            fade_step = 1.0 / (self.CROSSFADE_DURATION * self.sample_rate)
            chunk_fade_len = len(t_tensor)

            # Fade Out A (1.0 -> 0.0)
            # Fade In B (0.0 -> 1.0)
            # Note: This primitive logic assumes linear fade for simplicity
            start_fade = self.fade_progress
            end_fade = min(1.0, start_fade + (chunk_fade_len * fade_step))

            fade_curve = np.linspace(start_fade, end_fade, chunk_fade_len)

            mixed_audio = (audio_a * (1.0 - fade_curve)) + (audio_b * fade_curve)

            # Update state
            self.fade_progress = end_fade
            if self.fade_progress >= 1.0:
                # Swap Complete
                self.current_song_func = self.next_song_func
                self.next_song_func = None
                self.is_fading = False
                print("Hot-swap complete.")

            return mixed_audio

        return audio_a


# ==========================================
# 3. User Code Example (Conceptual)
# ==========================================
# content of "my_live_set.py"

"""
def song(t, state):
    # 't' is a vector of time for the current chunk
    # 'state' is a dict compatible object

    # 1. Tempo Sync
    bpm = 130
    beat = t * (bpm / 60.0)

    # 2. Access Persistent Counter
    # If we didn't use state, 'counter' would reset every chunk!
    # But here we just derive everything from 'beat' (Time),
    # so we might NOT need explicit state for patterns.
    # Statelesness is preferred for Time!

    kick_trig = (beat % 1.0) < 0.1

    # 3. State is useful for "Latching"
    # e.g. "Run this random seed only once"
    curr_seed = state.get_or_set("seed", 42)

    return Osc.Sine(440) * kick_trig
"""
```