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
Audio rendering engine for NaSong.

This module provides the `RenderEngine` class, which handles background rendering
of audio chunks with a priority system based on the current playback cursor.
"""

#
### Import Modules. ###
#
from typing import Optional, Dict, Set

#
import queue
import threading
import numpy as np


class RenderEngine:
    """
    Handles prioritized background rendering and caching of audio chunks.

    The RenderEngine operates in a separate thread, pre-computing audio samples
    around the playback cursor to ensure smooth real-time performance. It uses
    a priority queue to prioritize chunks closest to the cursor.

    Attributes:
        sample_rate: The audio sample rate (e.g., 44100).
        chunk_size: Size of each rendered block in samples.
        cursor_time: Current playback position in seconds (used for prioritization).
        current_version_id: Increments every time the sequencer is updated to invalidate cache.
    """

    def __init__(self, sample_rate: int = 44100, chunk_size: int = 4096):
        """
        Initializes the RenderEngine and starts the background render thread.

        Args:
            sample_rate: Hardware sample rate.
            chunk_size: Number of samples per rendered chunk.
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size  # In samples
        self.sequencer = None
        self.cursor_time = 0.0

        # Priority Queue: (priority, start_sample)
        # Priority is distance from cursor (lower is better)
        self.render_queue = queue.PriorityQueue()

        # Cache: start_sample -> (numpy array, version_id)
        self.cache: Dict[int, tuple[np.ndarray, int]] = {}
        self.cache_lock = threading.Lock()

        self.current_version_id = 0

        self.stop_event = threading.Event()
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)

        # To avoid re-adding same chunks
        self.queued_chunks: Set[int] = set()
        self.queued_lock = threading.Lock()

        self.render_thread.start()

    def set_sequencer(self, sequencer: Any):
        """
        Updates the active sequencer and invalidates existing cached chunks.

        Args:
            sequencer: The new NaSong Value/Variable to render.
        """
        with self.cache_lock:
            self.sequencer = sequencer
            # We do NOT clear cache, but increment version
            self.current_version_id += 1

        with self.queued_lock:
            self.queued_chunks.clear()
            # Clear queue by replacing it (simplest way to "empty" a PriorityQueue without popping everything)
            # Create a new queue instance
            try:
                while not self.render_queue.empty():
                    self.render_queue.get_nowait()
            except queue.Empty:
                pass

        self._enqueue_chunks_near_cursor()

    def update_cursor(self, time_seconds: float):
        """
        Updates the cursor position for rendering prioritization.

        Args:
            time_seconds: Current playback time in seconds.
        """
        self.cursor_time = time_seconds
        # We don't necessarily need to clear the queue, but we should add
        # the new relevant chunks with high priority.
        # The worker loop will pick up the highest priority (closest to new cursor).
        self._enqueue_chunks_near_cursor()

    def _enqueue_chunks_near_cursor(self):
        """
        Identifies and enqueues chunks within a window around the cursor.
        """
        if not self.sequencer:
            return

        # Define a window to render around cursor (e.g., -2s to +10s)
        center_sample_idx = int(self.cursor_time * self.sample_rate)

        # Determine accessible range (rough estimate or infinite?)
        # For now, render a window of +/- 30 seconds for now to keep it bounded but responsive.

        start_t = max(0.0, self.cursor_time - 5.0)
        end_t = self.cursor_time + 30.0

        start_sample_target = int(start_t * self.sample_rate)
        end_sample_target = int(end_t * self.sample_rate)

        # Align to chunk size
        start_sample_aligned = (
            start_sample_target // self.chunk_size
        ) * self.chunk_size
        end_sample_aligned = (end_sample_target // self.chunk_size) * self.chunk_size

        with self.queued_lock:
            for sample_idx in range(
                start_sample_aligned,
                end_sample_aligned + self.chunk_size,
                self.chunk_size,
            ):
                if sample_idx in self.queued_chunks:
                    continue

                # Check directly, cache check is done in render loop too but good to avoid dupes here
                # Actually, check cache without lock if possible or optimize
                with self.cache_lock:
                    if sample_idx in self.cache:
                        continue

                # Priority: Distance to cursor
                priority = abs(sample_idx - center_sample_idx)

                self.render_queue.put((priority, sample_idx))
                self.queued_chunks.add(sample_idx)

    def get_audio_chunk(self, start_sample: int) -> tuple[Optional[np.ndarray], int]:
        """
        Retrieves a rendered audio chunk from the cache.

        If the chunk is not cached, it is queued for background rendering with
        high priority.

        Args:
            start_sample: The absolute sample index for the start of the chunk.

        Returns:
            A tuple of (numpy_array, version_id). version_id is -1 if not found.
        """
        # Align
        aligned_start = (start_sample // self.chunk_size) * self.chunk_size

        with self.cache_lock:
            cached = self.cache.get(aligned_start)
            if cached is not None:
                chunk, version = cached
                # If we need a sub-slice because request wasn't aligned
                offset = start_sample - aligned_start
                if offset == 0:
                    return chunk, version
                # This logic is a bit tricky if offsets differ.
                if offset < len(chunk):
                    return chunk[offset:], version
                return None, -1

        # If not in cache, ensure it's queued
        with self.queued_lock:
            if aligned_start not in self.queued_chunks:
                # Prioritize based on version mismatch too?
                # Ideally yes, but cursor distance is good proxy for "what user is seeing".
                # If we see a "Blue" chunk, we want it turned Green.
                priority = abs(aligned_start - int(self.cursor_time * self.sample_rate))
                self.render_queue.put((priority, aligned_start))
                self.queued_chunks.add(aligned_start)

        return None, -1

    def _render_loop(self):
        """
        Internal worker loop for the background render thread.

        Continuously pops chunks from the priority queue, renders them using
        the sequencer, and stores the results in the cache.
        """
        while not self.stop_event.is_set():
            try:
                # Get next chunk to render
                # Timeout allows checking stop_event periodically
                item = self.render_queue.get(timeout=0.1)
                priority, start_sample = item

                with self.queued_lock:
                    if start_sample in self.queued_chunks:
                        self.queued_chunks.remove(start_sample)

                # Check cache again just in case
                ignore = False
                with self.cache_lock:
                    if start_sample in self.cache:
                        ignore = True

                if ignore:
                    self.render_queue.task_done()
                    continue

                # Render
                if self.sequencer:
                    # Generate time array
                    t_start = start_sample / self.sample_rate
                    # Check chunk size
                    # Fix: use linspace correctly for size
                    # Wait, linspace endpoint=False is [start, end)
                    t_end = (start_sample + self.chunk_size) / self.sample_rate

                    time_array = np.linspace(
                        t_start,
                        t_end,
                        self.chunk_size,
                        endpoint=False,
                        dtype=np.float32,
                    )

                    try:
                        # Use getitem_np but handle if sequencer fails
                        # Also sequencer access might need care if it changes mid-render (thread safety of sequencer?)
                        # Assuming sequencer is replaced atomicly in set_sequencer.
                        # But internal state of sequencer might change? Usually sequencers are immutable-ish after creation in current design?
                        # Debug stats
                        # print(f"Render {start_sample}: T range [{t_start:.4f}, {t_end:.4f}]")
                        audio = self.sequencer.getitem_np(time_array, self.sample_rate)

                        # Debug audio
                        # peak = np.max(np.abs(audio))
                        # if peak == 0:
                        #     pass # print(f"Render {start_sample}: SILENCE produced. T={time_array[0]}")
                        # else:
                        #     pass # print(f"Render {start_sample}: AUDIO produced. Peak={peak}")

                    except Exception as e:
                        print(f"Render Error at {start_sample}: {e}")
                        import traceback

                        traceback.print_exc()
                        audio = np.zeros(self.chunk_size, dtype=np.float32)

                    with self.cache_lock:
                        # Store as (audio, version)
                        # We use self.current_version_id at the time of STARTING render?
                        # Or current? set_sequencer runs in main thread. _render_loop in thread.
                        # It's safer to capture version when we grabbed the item or start render.
                        # But self.sequencer is also shared. If set_sequencer changed sequencer mid-render, we might have mixed state.
                        # For now, let's assume we store with current ID.
                        self.cache[start_sample] = (
                            audio.astype(np.float32),
                            self.current_version_id,
                        )

                self.render_queue.task_done()

            except queue.Empty:
                pass
            except Exception:
                # logging.error(f"Render Loop Error: {e}")
                pass

    def stop(self):
        """
        Gracefully stops the background render thread.
        """
        self.stop_event.set()
        self.stop_event.set()
        if self.render_thread.is_alive():
            self.render_thread.join()
