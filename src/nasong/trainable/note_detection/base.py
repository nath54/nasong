from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class NoteDetector(ABC):
    """
    Abstract base class for note detection algorithms.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the detector with configuration parameters.

        Args:
            config: Dictionary containing method-specific parameters.
        """
        self.config = config

    @abstractmethod
    def detect(self, audio_data: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
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
                    'frequencies': List[float], # List of frequencies in Hz (for polyphony)
                    'confidence': float,   # Optional confidence score
                },
                ...
            ]
        """
        pass
