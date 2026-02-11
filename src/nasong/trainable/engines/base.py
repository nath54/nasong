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
from numpy.typing import NDArray

#
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
    def step(self) -> dict[str, float]:
        """
        Performs a single optimization step (backward + update).

        Returns:
            A dictionary of metrics (e.g., {"loss": 0.5, "lr": 0.001}).
        """
        pass

    @abstractmethod
    def get_parameter_values(self) -> dict[str, float]:
        """
        Retrieves the current values of all trainable parameters.
        """
        pass

    @abstractmethod
    def set_parameter_values(self, parameters: dict[str, float]) -> None:
        """
        Injects values into the trainable parameters.
        """
        pass
