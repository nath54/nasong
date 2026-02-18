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


"""Base interface for music transcription and note detection backends.

Defines the abstract `NoteDetector` class which all specific detection
algorithms (Librosa, CREPE, Basic Pitch, etc.) must implement to be
compatible with the NaSong training pipeline.
"""

#
### Import Modules. ###
#
from typing import Any

#
from abc import ABC, abstractmethod

#
import numpy as np


#
class NoteDetector(ABC):
    """Abstract base class for note detection and transcription algorithms.

    Provides a consistent interface for extracting musical events (notes) from
    raw audio data, regardless of the underlying detection technology.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the detector with configuration parameters.

        Args:
            config: Dictionary containing method-specific parameters.
        """
        self.config = config

    @abstractmethod
    def detect(self, audio_data: np.ndarray, sample_rate: int) -> list[dict[str, Any]]:
        """
        Detect notes in the given audio segment.

        Args:
            audio_data: Audio data as numpy array (float32).
            sample_rate: Sample rate of the audio data.

        Returns:
            List of dictionaries, each representing a detected note event.
            Expected format:
            [
                {
                    'start_time': float,  # Start time in seconds
                    'duration': float,    # Duration in seconds
                    'frequencies': list[float], # List of frequencies in Hz (for polyphony)
                    'confidence': float,   # Optional confidence score
                },
                ...
            ]
        """
        pass  # pylint: disable=unnecessary-pass
