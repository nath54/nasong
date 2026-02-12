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
Random Float Value implementation.

This module provides the `RandomFloat` class, which represents a Value that
returns a random float within a specified range for each sample. The range
boundaries are themselves `Value` objects, allowing for dynamic range control.

Example:
    >>> from nasong.core.values.basic.value_random_float import RandomFloat
    >>> from nasong.core.values.basic.value_constant import Constant, c
    >>> val = RandomFloat(min_range=c(0.0), max_range=c(1.0))
    >>> 0.0 <= val.get_item(0, 44100) <= 1.0
    True
"""

#
### Import Modules. ###
#
import random

#
import numpy as np
from numpy.typing import NDArray

from typing import Any

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor


#
class RandomFloat(Value):
    """A Value that returns a random float within a specified range.

    This class generates stochastic values by sampling from a uniform
    distribution defined by two boundary Value objects.

    Attributes:
        min_range (Value): The lower bound Value.
        max_range (Value): The upper bound Value.
    """

    #
    def __init__(self, min_range: Value, max_range: Value) -> None:
        """Initializes the RandomFloat Value.

        Args:
            min_range (Value): The Value defining the lower bound.
            max_range (Value): The Value defining the upper bound.
        """
        super().__init__()
        self.min_range: Value = min_range
        self.max_range: Value = max_range

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns a random float for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: A random float within [min, max].
        """
        return random.uniform(
            a=float(self.min_range.get_item(index=index, sample_rate=sample_rate)),
            b=float(self.max_range.get_item(index=index, sample_rate=sample_rate)),
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized array of random floats.

        This override provides a performance-critical implementation using
        NumPy's vectorized random number generator.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: An array of random floats.
        """

        #
        ### Get the vectorized min and max boundaries. ###
        #
        min_vals: NDArray[np.float32] = self.min_range.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        max_vals: NDArray[np.float32] = self.max_range.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        ### Use numpy's vectorized uniform random number generator. ###
        #
        random_vals: NDArray[np.float32] = np.random.uniform(
            low=0.0, high=1.0, size=indexes_buffer.shape
        ).astype(np.float32)

        # Save for backward pass
        self._last_random_vals = random_vals

        return min_vals + random_vals * (max_vals - min_vals)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Returns differentiable random floats for training.

        The min and max boundaries are treated as differentiable inputs,
        allowing the range to be learned during training.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of random floats.
        """

        #
        ### Get the vectorized min and max boundaries. ###
        #
        min_vals: Tensor = self.min_range.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        max_vals: Tensor = self.max_range.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        ### Use uniform random distribution with trainable bounds. ###
        #
        random_vals: Tensor = torch.rand_like(
            indexes_buffer, dtype=torch.float32, device=device
        )
        #
        return min_vals + random_vals * (max_vals - min_vals)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients to min_range and max_range.

        Uses the saved random values to compute the partial derivatives
        with respect to the boundaries:
        y = min + r*(max - min) = min*(1-r) + max*r
        dy/dmin = 1 - r
        dy/dmax = r

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        if hasattr(self, "_last_random_vals"):
            r = self._last_random_vals
            self.min_range.backward(grad_output * (1.0 - r), context, sample_rate)
            self.max_range.backward(grad_output * r, context, sample_rate)
