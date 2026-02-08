#
### Engine Abstraction Layer. ###
#

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from numpy.typing import NDArray

# Internal imports
from nasong.core.value import Value


class BaseTrainingEngine(ABC):
    """
    Abstract base class for all training engines (Torch, NumPy, Autograd).

    An engine is responsible for:
    1. Running the forward pass to compute loss.
    2. Calculating gradients for all capture parameters.
    3. Applying parameter updates via an optimizer.
    """

    @abstractmethod
    def __init__(self, config: Any) -> None:
        """
        Initializes the engine with a configuration object.
        """
        self.config = config

    @abstractmethod
    def compute_loss(
        self, target_audio: NDArray[np.float32], blueprint: Value, sample_rate: int
    ) -> float:
        """
        Runs the model and calculates the loss against target audio.

        Args:
            target_audio: The ground truth audio data.
            blueprint: The root Value node of the audio graph.
            sample_rate: The sample rate for rendering.

        Returns:
            The scalar loss value.
        """
        pass

    @abstractmethod
    def step(self) -> Dict[str, float]:
        """
        Performs a single optimization step (backward + update).

        Returns:
            A dictionary of metrics (e.g., {"loss": 0.5, "lr": 0.001}).
        """
        pass

    @abstractmethod
    def get_parameter_values(self) -> Dict[str, float]:
        """
        Retrieves the current values of all trainable parameters.
        """
        pass

    @abstractmethod
    def set_parameter_values(self, parameters: Dict[str, float]) -> None:
        """
        Injects values into the trainable parameters.
        """
        pass
