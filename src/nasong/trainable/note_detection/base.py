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
from abc import ABC, abstractmethod

#
import numpy as np


#
class NoteDetector(ABC):
    """
    Abstract base class for note detection algorithms.
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
        pass
