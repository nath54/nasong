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


"""Classic DSP note detection backend using Librosa.

Implements transcription by combining energy-based onset detection with
probabilistic YIN (pYIN) pitch tracking. Robust for monophonic sources
and does not require neural network models.
"""

#
### Import Modules. ###
#
from typing import Any

#
import numpy as np

#
from .base import NoteDetector

#
### Try to import librosa ###
#
try:
    import librosa
except ImportError:
    librosa = None  # type: ignore


#
class LibrosaDetector(NoteDetector):
    """Traditional DSP-based note detection using Librosa.

    Uses `librosa.onset.onset_detect` for event boundaries and
    `librosa.pyin` for pitch estimation within those boundaries.
    """

    def detect(self, audio_data: np.ndarray, sample_rate: int) -> list[dict[str, Any]]:
        """Detects notes using onset detection followed by pYIN.

        Args:
            audio_data (np.ndarray): Raw audio samples.
            sample_rate (int): Audio sampling rate.

        Returns:
            list[dict[str, Any]]: Detected notes with pitch, timing, and
                confidence metrics.
        """
        if librosa is None:
            raise ImportError(
                "Librosa is not installed. Please install it to use this detector."
            )

        fmin = self.config.get("librosa_fmin", 50.0)
        fmax = self.config.get("librosa_fmax", 2000.0)
        frame_len = self.config.get("librosa_frame_length", 2048)
        hop_len = self.config.get("librosa_hop_length", 512)

        # 1. Run pYIN on the whole audio to capture full temporal context
        # Handle cases where the audio is shorter than the frame length
        if len(audio_data) < frame_len:
            audio_padded = np.pad(audio_data, (0, frame_len - len(audio_data)), mode="constant")
        else:
            audio_padded = audio_data

        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_padded,
            sr=sample_rate,
            fmin=fmin,
            fmax=fmax,
            frame_length=frame_len,
            hop_length=hop_len,
        )

        # Precompute the times corresponding to each frame in the whole-audio pitch track
        frame_times = librosa.frames_to_time(
            np.arange(len(f0)), sr=sample_rate, hop_length=hop_len
        )

        # 2. Run Onset detection on the entire audio
        onset_frames = librosa.onset.onset_detect(
            y=audio_data, sr=sample_rate, backtrack=True, units="frames"
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)

        # If no onsets, detect pitch on whole segment
        if len(onset_times) == 0:
            times = np.array([0.0])
        else:
            times = onset_times

        notes = []
        total_duration = len(audio_data) / sample_rate

        # 3. Segment the precomputed pitches by onset intervals
        for i, start_t in enumerate(times):
            end_t = times[i + 1] if i < len(times) - 1 else total_duration
            duration = end_t - start_t

            if duration < 0.02:  # Skip extremely short transient glitches
                continue

            # Find frames that fall within the current onset interval
            in_segment = (frame_times >= start_t) & (frame_times < end_t)
            segment_voiced_flag = voiced_flag[in_segment]
            segment_f0 = f0[in_segment]
            segment_probs = voiced_probs[in_segment]

            voiced_f0 = segment_f0[segment_voiced_flag]

            if len(voiced_f0) > 0:
                pitch = np.median(voiced_f0)
                confidence = float(np.mean(segment_probs[segment_voiced_flag]))
            else:
                # If no voiced pitch is found, skip this segment
                continue

            # Calculate actual RMS amplitude of the segment from original audio
            start_sample = int(start_t * sample_rate)
            end_sample = int(end_t * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            amplitude = (
                float(np.sqrt(np.mean(segment_audio**2)))
                if len(segment_audio) > 0
                else 0.0
            )

            notes.append(
                {
                    "start_time": float(start_t),
                    "duration": float(duration),
                    "frequencies": [float(pitch)],
                    "confidence": confidence,
                    "amplitude": amplitude,
                }
            )

        return notes

