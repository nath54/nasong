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
Random Choice Value implementation.

This module provides the `RandomChoice` class, which represents a Value that
randomly selects from a list of other `Value` objects. This is useful for
stochastic variations in audio synthesis.

Example:
    >>> from nasong.core.values.basic.value_random_choice import RandomChoice
    >>> from nasong.core.values.basic.value_constant import Constant
    >>> v1, v2 = Constant(100.0), Constant(200.0)
    >>> val = RandomChoice(choices=[v1, v2])
    >>> # returns 100.0 or 200.0 randomly
    >>> val.get_item(0, 44100) in [100.0, 200.0]
    True
"""

#
### Import Modules. ###
#
from typing import Any
import random
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor


#
class RandomChoice(Value):
    """A Value that randomly selects from a list of other Value objects.

    This class provides a way to introduce stochasticity by picking one
    Value from a provided list for each sample index (or the first one
    consistently for vectorized operations to maintain stability).

    Attributes:
        choices (list[Value]): A list of Value objects to choose from.
    """

    #
    def __init__(self, choices: list[Value]) -> None:
        """Initializes the RandomChoice Value.

        Args:
            choices (list[Value]): A list of candidate Value objects.
        """
        super().__init__()
        self.choices: list[Value] = choices

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns a random item from the choices for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: A sample from a randomly selected choice, or 0.0 if empty.
        """
        if not self.choices:
            return 0.0

        return random.choice(self.choices).get_item(
            index=index, sample_rate=sample_rate
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a NumPy array from the first choice.

        For NumPy rendering, we choose the first choice to maintain consistency
        with Torch and ensure stable gradients/output during training/batching.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: An array of values from the first choice.
        """
        if not self.choices:
            return np.zeros_like(indexes_buffer)

        return self.choices[0].getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Returns a PyTorch tensor from the first choice for training.

        The selection is fixed to the first choice to maintain stable gradient
        flow and consistency across training steps.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of values from the first choice.
        """
        if len(self.choices) > 0:
            return self.choices[0].getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            )
        else:
            return torch.zeros_like(indexes_buffer, dtype=torch.float32, device=device)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradient to the first choice.

        Consistent with `getitem_np` and `getitem_torch`, only the first
        choice receives gradients.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        if self.choices:
            self.choices[0].backward(grad_output, context, sample_rate)
