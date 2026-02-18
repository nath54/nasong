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


"""Note detection backend using the CREPE model via ONNX Runtime.

Implements high-resolution monophonic pitch tracking using the 'tiny'
CREPE model serialized to ONNX. This allows for neural-network based
transcription without a full deep learning framework dependency.
"""

#
### Import Modules. ###
#
from typing import Any, Optional

#
import os
import urllib.request
import numpy as np

#
from .base import NoteDetector

#
### Try importing onnxruntime, handle failure gracefully ###
#
try:
    import onnxruntime as ort  # type: ignore

    HAS_ORT = True
except ImportError:
    HAS_ORT = False

#
### Try importing scipy for resampling ###
#
try:
    from scipy import signal  # type: ignore

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Constants for CREPE
CREPE_MODEL_URL = (
    "https://github.com/yqzhishen/onnxcrepe/releases/download/v1.1.0/tiny.onnx"
)


#
class OnnxCrepeDetector(NoteDetector):
    """Neural-network based pitch detection using CREPE with ONNX Runtime.

    Handles model downloading, audio resampling to 16kHz, framing,
    inference, and conversion of frame-level activations to note events.
    """

    def detect(self, audio_data: np.ndarray, sample_rate: int) -> list[dict[str, Any]]:
        """Transcribes audio using the CREPE ONNX model.

        Args:
            audio_data (np.ndarray): Raw audio signal.
            sample_rate (int): Input sampling rate.

        Returns:
            list[dict[str, Any]]: Transcribed note events.
        """
        if not HAS_ORT:
            raise ImportError(
                "onnxruntime is required for OnnxCrepeDetector. Please install 'onnxruntime'."
            )

        # 1. Ensure Model Exists
        model_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "assets", "models"
        )
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "crepe.onnx")

        if not os.path.exists(model_path):
            print(f"Downloading CREPE ONNX model to {model_path}...")
            # Using the huggingface link found
            try:
                urllib.request.urlretrieve(CREPE_MODEL_URL, model_path)
                print("Download complete.")
            except Exception as e:  # pylint: disable=broad-except
                raise RuntimeError(
                    f"Failed to download CREPE model from {CREPE_MODEL_URL}. Error: {e}"
                ) from e

        # 2. Preprocessing
        # CREPE requires 16kHz audio
        target_sr = 16000
        if sample_rate != target_sr:
            if HAS_SCIPY:
                # Calculate number of samples
                num_samples = int(len(audio_data) * target_sr / sample_rate)
                audio_16k = signal.resample(audio_data, num_samples)
            else:
                # Simple linear interpolation fallback
                x_old = np.linspace(0, 1, len(audio_data))
                x_new = np.linspace(
                    0, 1, int(len(audio_data) * target_sr / sample_rate)
                )
                audio_16k = np.interp(x_new, x_old, audio_data)
        else:
            audio_16k = audio_data

        # Frame the audio
        # CREPE uses 1024 sample window, hop size usually 10ms (160 samples)
        window_size = 1024
        hop_length = 160  # 10ms at 16kHz

        # Pad audio to center frames
        pad_width = window_size // 2
        audio_padded = np.pad(audio_16k, (pad_width, pad_width), mode="constant")

        num_frames = (len(audio_padded) - window_size) // hop_length + 1

        # Generator to avoid massive memory usage for frames
        def frame_generator():
            for i in range(num_frames):
                start = i * hop_length
                end = start + window_size
                frame = audio_padded[start:end]

                # Normalize frame
                frame -= np.mean(frame)
                std = np.std(frame)
                if std > 1e-6:
                    frame /= std

                yield frame

        frames = np.array(list(frame_generator()), dtype=np.float32)

        # ONNX Inference
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        # Run in batches to avoid OOM
        batch_size = 128
        activation_list = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            outputs = session.run(None, {input_name: batch})
            activation_list.append(outputs[0])

        activations = np.concatenate(
            activation_list, axis=0
        )  # Shape: (num_frames, 360)

        # 3. Postprocessing
        confidence_threshold = self.config.get("crepe_confidence_threshold", 0.6)

        # Convert activations to pitch/confidence
        # CREPE mapping: cents = 20 * bin + 1997.3794084376191
        # frequency = 10 * 2 ** (cents / 1200)

        pitches = []
        confidences = []

        for act in activations:
            best_bin = int(np.argmax(act))
            confidence = act[best_bin]

            # Optional: weighted average for better precision
            # Use local window around peak
            start = max(0, best_bin - 4)
            end = min(360, best_bin + 5)
            weights = act[start:end]
            bins = np.arange(start, end)

            if np.sum(weights) > 0:
                avg_bin = np.sum(weights * bins) / np.sum(weights)
                cents = 20 * avg_bin + 1997.3794084376191
                freq = 10 * (2 ** (cents / 1200))
            else:
                freq = 0.0

            pitches.append(freq)
            confidences.append(confidence)

        pitches_arr = np.array(pitches)
        confidences_arr = np.array(confidences)

        # 4. Convert frames to note events
        # This is a simplified "monophonic" tracker logic
        # merge consecutive frames above threshold

        notes: list[dict[str, Any]] = []
        current_note: Optional[dict[str, Any]] = None

        # Time per frame in original audio seconds
        dt = hop_length / 16000.0

        for i, (freq, conf) in enumerate(zip(pitches_arr, confidences_arr)):
            time = i * dt

            if conf >= confidence_threshold:
                if current_note is None:
                    # Start new note
                    current_note = {
                        "start_time": time,
                        "freqs": [freq],
                        "confs": [conf],
                    }
                else:
                    # Continue note
                    # Break if pitch changes too drastically (e.g. > semi-tone)
                    # For now simplistically, just append
                    if (
                        abs(freq - current_note["freqs"][-1])
                        > current_note["freqs"][-1] * 0.06
                    ):  # approx semitone
                        # End current, start new
                        self._finish_note(notes, current_note, audio_data, sample_rate)
                        current_note = {
                            "start_time": time,
                            "freqs": [freq],
                            "confs": [conf],
                        }
                    else:
                        current_note["freqs"].append(freq)
                        current_note["confs"].append(conf)
            else:
                if current_note is not None:
                    # End note
                    self._finish_note(notes, current_note, audio_data, sample_rate)
                    current_note = None

        if current_note is not None:
            self._finish_note(notes, current_note, audio_data, sample_rate)

        return notes

    def _finish_note(self, notes_list, note_data, audio_data, sample_rate):
        duration = len(note_data["freqs"]) * (160 / 16000.0)
        if duration < 0.05:  # Minimum duration
            return

        avg_freq = np.median(note_data["freqs"])
        avg_conf = np.mean(note_data["confs"])

        # Calculate Amplitude from original audio
        start_sample = int(note_data["start_time"] * sample_rate)
        end_sample = int((note_data["start_time"] + duration) * sample_rate)
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)

        amplitude = 0.5
        if end_sample > start_sample:
            segment = audio_data[start_sample:end_sample]
            amplitude = float(np.sqrt(np.mean(segment**2)))

        notes_list.append(
            {
                "start_time": float(note_data["start_time"]),
                "duration": float(duration),
                "frequencies": [float(avg_freq)],
                "confidence": float(avg_conf),
                "amplitude": float(amplitude),
            }
        )
