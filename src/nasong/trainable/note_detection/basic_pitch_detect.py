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


"""Note detection backend using Spotify's Basic Pitch.

Uses the `basic-pitch` library for polyphonic piano transcription and
general-purpose instrument note detection. It typically runs via ONNX
for efficiency and to avoid heavy deep learning dependencies.
"""

#
### Import Modules. ###
#
from typing import Any

#
import os
import tempfile
import numpy as np

#
from .base import NoteDetector

#
### Try to import basic_pitch and soundfile. ###
#
try:
    from basic_pitch.inference import predict  # type: ignore
    import soundfile as sf  # type: ignore
except ImportError:
    predict = None  # type: ignore
    sf = None  # type: ignore


#
class BasicPitchDetector(NoteDetector):
    """Note detection using Spotify's Basic Pitch model.

    Particularly effective for polyphonic audio and complex instrument
    recordings. Interacts with the `predict` function from the
    `basic-pitch` library.
    """

    def detect(self, audio_data: np.ndarray, sample_rate: int) -> list[dict[str, Any]]:
        """Detects notes using Basic Pitch inference.

        Args:
            audio_data (np.ndarray): The audio to transcribe.
            sample_rate (int): Sampling rate of the audio.

        Returns:
            list[dict[str, Any]]: List of note event dictionaries including
                pitch, timing, and confidence.
        """
        if predict is None:
            raise ImportError(
                "Basic Pitch or SoundFile is not installed. Please install 'basic-pitch' and 'soundfile'."
            )

        # Write to temp file because basic-pitch expects a file path
        # (predict takes a path string)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_data, sample_rate)
            tmp_path = tmp.name

        try:
            # Run inference
            # basic-pitch handles downloading the model if not present.
            # We enforce ONNX serialization to avoid TensorFlow dependency.
            # predict expects a list of paths
            # basic-pitch 0.4.0 takes single audio_path, no model_serialization arg
            _model_output, _midi_data, note_events = predict(
                tmp_path,
                onset_threshold=self.config.get("bp_onset_threshold", 0.5),
                frame_threshold=self.config.get("bp_frame_threshold", 0.3),
                minimum_note_length=self.config.get("bp_min_note_len", 58.0),
                minimum_frequency=self.config.get("bp_min_freq", 50.0),
                maximum_frequency=self.config.get("bp_max_freq", 2000.0),
            )

            if not note_events:
                return []

            # note_events is returned directly as list of tuples in 0.4.0
            file_events = note_events

        except Exception as e:  # pylint: disable=broad-except
            # Check if basic-pitch error is related to missing ONNX runtime
            if "onnxruntime" in str(e).lower():
                raise ImportError(
                    "ONNX Runtime is required for Basic Pitch ONNX mode. Please install 'onnxruntime' or 'onnxruntime-gpu'."
                ) from e
            raise e
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        notes = []
        # note_events is a list of lists of tuples
        for start, end, pitch_midi, amp, _bends in file_events:
            # Convert MIDI pitch to Hz
            freq = 440.0 * (2.0 ** ((pitch_midi - 69.0) / 12.0))

            notes.append(
                {
                    "start_time": float(start),
                    "duration": float(end - start),
                    "frequencies": [float(freq)],
                    "confidence": float(amp),
                    "amplitude": float(amp),
                }
            )

        # Basic Pitch might return overlapping notes (polyphony).
        # We process them individually.
        return notes
