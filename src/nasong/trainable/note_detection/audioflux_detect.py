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
### Try to import audioflux. ###
#
try:
    import audioflux as af
except ImportError:
    af = None


class AudioFluxDetector(NoteDetector):
    """
    Note detection using AudioFlux (PEF or YIN algorithms).
    High-performance C/C++ core with Python wrapper.
    Typically monophonic.
    """

    def detect(
        self, audio_segment: np.ndarray, sample_rate: int
    ) -> List[Dict[str, Any]]:
        if af is None:
            raise ImportError("AudioFlux is not installed. Please install 'audioflux'.")

        # AudioFlux expects contiguous C-order array
        audio_segment = np.ascontiguousarray(audio_segment, dtype=np.float32)

        algo_type = self.config.get("audioflux_type", "PEF")
        min_freq = self.config.get("audioflux_min_freq", 50.0)
        max_freq = self.config.get("audioflux_max_freq", 2000.0)
        slide_length = self.config.get("audioflux_slide_length", 1024)
        radix2_exp = self.config.get("audioflux_radix2_exp", 12)  # 2^12 = 4096

        # Pitch processing
        # Note: AudioFlux Pitch API varies by algorithm

        pitches = []
        times = []
        _confidences = []  # PEF might not return confidence directly in same way as YIN

        if algo_type == "YIN":
            # YIN implementation in AudioFlux
            pitch_obj = af.PitchYIN(
                samplate=sample_rate,
                low_fre=min_freq,
                high_fre=max_freq,
                radix2_exp=radix2_exp,  # 2^12 = 4096
                slide_length=slide_length,
            )
            fre_arr, val1_arr, val2_arr = pitch_obj.pitch(audio_segment)

            pitches = fre_arr
            num_frames = len(fre_arr)
            times = np.arange(num_frames) * (slide_length / sample_rate)
            _confidences = val2_arr  # approximation

        else:  # PEF
            pitch_obj = af.PitchPEF(
                samplate=sample_rate,
                low_fre=min_freq,
                high_fre=max_freq,
                radix2_exp=radix2_exp,
                slide_length=slide_length,
            )
            fre_arr = pitch_obj.pitch(audio_segment)

            pitches = fre_arr
            num_frames = len(fre_arr)
            times = np.arange(num_frames) * (slide_length / sample_rate)
            # PEF doesn't return confidence directly in pitch(), so we use dummy 1.0s or maybe magnitude if separate call
            _confidences = np.ones_like(fre_arr)
        # Segmentation Logic (Simple voicing check)
        notes = []
        current_start = None
        current_pitch_accum = []

        # Threshold for voicing. 0 means unvoiced usually in AF results
        for i, f in enumerate(pitches):
            if f > 0:  # Voiced
                if current_start is None:
                    current_start = times[i]
                current_pitch_accum.append(f)
            else:
                if current_start is not None:
                    # End note
                    duration = times[i] - current_start
                    if duration > 0.05:
                        mean_pitch = np.mean(current_pitch_accum)

                        # Calculate amplitude
                        start_sample = int(current_start * sample_rate)
                        end_sample = int(times[i] * sample_rate)
                        start_sample = max(0, start_sample)
                        end_sample = min(len(audio_segment), end_sample)

                        if end_sample > start_sample:
                            segment = audio_segment[start_sample:end_sample]
                            amplitude = float(np.sqrt(np.mean(segment**2)))
                        else:
                            amplitude = 0.0

                        notes.append(
                            {
                                "start_time": float(current_start),
                                "duration": float(duration),
                                "frequencies": [float(mean_pitch)],
                                "confidence": 1.0,
                                "amplitude": amplitude,
                            }
                        )
                    current_start = None
                    current_pitch_accum = []

        if current_start is not None:
            duration = times[-1] - current_start
            if duration > 0.05:
                mean_pitch = np.mean(current_pitch_accum)

                # Calculate amplitude
                start_sample = int(current_start * sample_rate)
                end_sample = int(times[-1] * sample_rate)
                start_sample = max(0, start_sample)
                end_sample = min(len(audio_segment), end_sample)

                if end_sample > start_sample:
                    segment = audio_segment[start_sample:end_sample]
                    amplitude = float(np.sqrt(np.mean(segment**2)))
                else:
                    amplitude = 0.0

                notes.append(
                    {
                        "start_time": float(current_start),
                        "duration": float(duration),
                        "frequencies": [float(mean_pitch)],
                        "confidence": 1.0,
                        "amplitude": amplitude,
                    }
                )

        return notes
