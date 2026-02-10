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
