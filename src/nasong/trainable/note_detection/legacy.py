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
TODO: add full docstring, explaining what the goal of this script is, and explaining for each class and each function what is it, how it works, and how to use it.
"""

#
### Import Modules. ###
#
from typing import List, Dict, Any

#
import numpy as np

#
from .base import NoteDetector


#
class LegacyDetector(NoteDetector):
    """
    Original energy-based onset detection and FFT pitch tracking.
    """

    def detect(
        self, audio_segment: np.ndarray, sample_rate: int
    ) -> List[Dict[str, Any]]:
        # Unpack config
        num_notes = self.config.get("legacy_num_notes", 1)
        use_onset_detection = self.config.get("legacy_use_onset", True)

        total_duration = len(audio_segment) / sample_rate
        notes = []

        if use_onset_detection:
            # Simple energy-based onset detection
            # Calculate short-time energy
            frame_len_sec = self.config.get("legacy_frame_len", 0.02)
            hop_len_sec = self.config.get("legacy_hop_len", 0.01)

            frame_length = int(frame_len_sec * sample_rate)
            hop_length = int(hop_len_sec * sample_rate)

            energy = []
            for i in range(0, len(audio_segment) - frame_length, hop_length):
                frame = audio_segment[i : i + frame_length]
                energy.append(np.sum(frame**2))

            energy = np.array(energy)

            # Find peaks in energy (onsets)
            threshold_ratio = self.config.get("legacy_onset_threshold", 0.3)
            threshold = threshold_ratio * np.max(energy)
            onsets = []
            for i in range(1, len(energy) - 1):
                if (
                    energy[i] > threshold
                    and energy[i] > energy[i - 1]
                    and energy[i] > energy[i + 1]
                ):
                    onset_time = i * hop_length / sample_rate
                    onsets.append(onset_time)

            # Limit to num_notes
            onsets = onsets[:num_notes]

            # If no onsets detected, fall back to simple division
            if len(onsets) == 0:
                onsets = [i * total_duration / num_notes for i in range(num_notes)]
        else:
            # Simple even division
            onsets = [i * total_duration / num_notes for i in range(num_notes)]

        # Extract pitch and duration for each onset
        for i, start_time in enumerate(onsets):
            # Determine duration (to next onset or end)
            if i < len(onsets) - 1:
                duration = onsets[i + 1] - start_time
            else:
                duration = total_duration - start_time

            # Extract segment for pitch detection
            start_idx = int(start_time * sample_rate)
            # Use first 100ms or duration, whichever is shorter
            analysis_duration = min(0.1, duration)
            end_idx = int((start_time + analysis_duration) * sample_rate)
            note_segment = audio_segment[start_idx:end_idx]

            # Detect multiple pitches (chord detection)
            max_pitches = self.config.get("legacy_max_pitches", 3)
            min_freq = self.config.get("legacy_min_freq", 50.0)
            max_freq = self.config.get("legacy_max_freq", 4000.0)

            frequencies = self._detect_pitches_fft(
                note_segment,
                sample_rate,
                max_pitches=max_pitches,
                min_freq=min_freq,
                max_freq=max_freq,
            )

            notes.append(
                {
                    "frequencies": frequencies,  # List of frequencies (chord)
                    "start_time": start_time,
                    "duration": duration * 0.9,  # Leave small gap
                    "amplitude": float(np.sqrt(np.mean(note_segment**2)))
                    if len(note_segment) > 0
                    else 0.0,
                }
            )

        return notes

    def _detect_pitches_fft(
        self,
        audio_segment: np.ndarray,
        sample_rate: int,
        max_pitches: int = 3,
        min_freq: float = 50.0,
        max_freq: float = 4000.0,
    ) -> List[float]:
        """
        Detect multiple pitches using FFT peak detection.
        """

        if len(audio_segment) == 0:
            return [220.0]  # Default A3

        # Apply window
        windowed = audio_segment * np.hanning(len(audio_segment))

        # FFT
        fft = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(windowed), 1 / sample_rate)
        magnitudes = np.abs(fft)

        # Focus on frequency range
        freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
        freqs_filtered = freqs[freq_mask]
        mags_filtered = magnitudes[freq_mask]

        if len(mags_filtered) == 0:
            return [220.0]

        # Find peaks
        detected_freqs = []
        threshold = 0.1 * np.max(mags_filtered)

        # Find local maxima
        for i in range(1, len(mags_filtered) - 1):
            if (
                mags_filtered[i] > threshold
                and mags_filtered[i] > mags_filtered[i - 1]
                and mags_filtered[i] > mags_filtered[i + 1]
            ):
                detected_freqs.append((freqs_filtered[i], mags_filtered[i]))

        # Sort by magnitude and take top max_pitches
        detected_freqs.sort(key=lambda x: x[1], reverse=True)
        detected_freqs = detected_freqs[:max_pitches]

        # Extract just frequencies, sorted by frequency
        result = [f for f, _ in detected_freqs]
        result.sort()

        if len(result) == 0:
            return [220.0]

        return result
