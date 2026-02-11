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
    librosa = None


#
class LibrosaDetector(NoteDetector):
    """
    Note detection using Librosa (onset detection + pyin).
    Robust for monophonic instruments.
    """

    def detect(
        self, audio_segment: np.ndarray, sample_rate: int
    ) -> list[dict[str, Any]]:
        if librosa is None:
            raise ImportError(
                "Librosa is not installed. Please install it to use this detector."
            )

        # Onset detection
        onset_frames = librosa.onset.onset_detect(
            y=audio_segment, sr=sample_rate, backtrack=True, units="frames"
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)

        # If no onsets, detect pitch on whole segment
        if len(onset_times) == 0:
            times = np.array([0.0])
        else:
            times = onset_times
            # Ensure 0.0 is included if first onset is late?
            # Usually onset detection finds the start.
            # If the note starts at 0, onset_detect might find 0 or not.
            pass

        notes = []
        total_duration = len(audio_segment) / sample_rate

        fmin = self.config.get("librosa_fmin", 50.0)
        fmax = self.config.get("librosa_fmax", 2000.0)
        frame_len = self.config.get("librosa_frame_length", 2048)
        hop_len = self.config.get("librosa_hop_length", 512)

        for i, start_t in enumerate(times):
            end_t = times[i + 1] if i < len(times) - 1 else total_duration
            duration = end_t - start_t

            # Extract audio
            start_sample = int(start_t * sample_rate)
            end_sample = int(end_t * sample_rate)

            # Simple check for very short segments
            if end_sample - start_sample < frame_len:
                continue

            segment = audio_segment[start_sample:end_sample]

            # Run pYIN
            f0, voiced_flag, voiced_probs = librosa.pyin(
                segment,
                sr=sample_rate,
                fmin=fmin,
                fmax=fmax,
                frame_length=frame_len,
                hop_length=hop_len,
            )

            # Filter unvoiced
            voiced_f0 = f0[voiced_flag]

            if len(voiced_f0) > 0:
                # Use median pitch
                pitch = np.median(voiced_f0)

                notes.append(
                    {
                        "start_time": float(start_t),
                        "duration": float(duration),
                        "frequencies": [float(pitch)],
                        "confidence": float(np.mean(voiced_probs[voiced_flag])),
                        "amplitude": float(np.sqrt(np.mean(segment**2)))
                        if len(segment) > 0
                        else 0.0,
                    }
                )

        return notes
