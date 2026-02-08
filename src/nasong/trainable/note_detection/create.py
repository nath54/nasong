import dataclasses
from .base import NoteDetector
from .legacy import LegacyDetector
from .basic_pitch_detect import BasicPitchDetector
from .librosa_detect import LibrosaDetector
from .torchcrepe_detect import TorchCrepeDetector
from .audioflux_detect import AudioFluxDetector
from .onnx_crepe_detect import OnnxCrepeDetector
from ..config import NoteDetectionConfig


def create_note_detector(config: NoteDetectionConfig) -> NoteDetector:
    """
    Factory function to create a note detector instance.
    """
    method = config.method.lower().replace("-", "_").strip()
    config_dict = dataclasses.asdict(config)

    if method == "legacy":
        return LegacyDetector(config_dict)
    elif method in ["basic_pitch_onnx", "basic_pitch", "neuralnote", "neural_note"]:
        # NeuralNote uses basic-pitch ONNX under the hood
        return BasicPitchDetector(config_dict)
    elif method == "librosa":
        return LibrosaDetector(config_dict)
    elif method == "torchcrepe":
        return TorchCrepeDetector(config_dict)
    elif method == "audioflux":
        return AudioFluxDetector(config_dict)
    elif method == "crepe_onnx":
        return OnnxCrepeDetector(config_dict)
    else:
        raise ValueError(f"Unknown note detection method: {method}")
