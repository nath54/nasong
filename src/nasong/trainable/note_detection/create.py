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


"""Factory module for instantiating note detectors.

Provides the `create_note_detector` function which maps configuration
settings to appropriate `NoteDetector` implementations.
"""

#
### Import Modules. ###
#
import dataclasses

#
from .base import NoteDetector
from .legacy import LegacyDetector
from .basic_pitch_detect import BasicPitchDetector
from .librosa_detect import LibrosaDetector
from .torchcrepe_detect import TorchCrepeDetector
from .audioflux_detect import AudioFluxDetector
from .onnx_crepe_detect import OnnxCrepeDetector
from ..config import NoteDetectionConfig


#
def create_note_detector(config: NoteDetectionConfig) -> NoteDetector:
    """Instantiates a note detector based on the provided configuration.

    Args:
        config (NoteDetectionConfig): The configuration object specifying
            the detection method and its hyperparameters.

    Returns:
        NoteDetector: An initialized instance of the requested detector.

    Raises:
        ValueError: If the requested detection method is unknown.
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
