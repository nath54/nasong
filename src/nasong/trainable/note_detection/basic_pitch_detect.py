from typing import List, Dict, Any
import numpy as np
import tempfile
import os
import logging
from .base import NoteDetector

try:
    from basic_pitch.inference import predict
    import soundfile as sf
except ImportError:
    predict = None
    sf = None


class BasicPitchDetector(NoteDetector):
    """
    Note detection using Basic Pitch (Spotify) with ONNX runtime.
    """

    def detect(
        self, audio_segment: np.ndarray, sample_rate: int
    ) -> List[Dict[str, Any]]:
        if predict is None:
            raise ImportError(
                "Basic Pitch or SoundFile is not installed. Please install 'basic-pitch' and 'soundfile'."
            )

        # Write to temp file because basic-pitch expects a file path
        # (predict takes a path string)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_segment, sample_rate)
            tmp_path = tmp.name

        try:
            # Run inference
            # basic-pitch handles downloading the model if not present.
            # We enforce ONNX serialization to avoid TensorFlow dependency.
            # predict expects a list of paths
            # basic-pitch 0.4.0 takes single audio_path, no model_serialization arg
            model_output, midi_data, note_events = predict(
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

        except Exception as e:
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
        for start, end, pitch_midi, amp, bends in file_events:
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
