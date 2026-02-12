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
Random Integer Value implementation.

This module provides the `RandomInt` class, which represents a Value that
returns a random integer within a specified range for each sample. The range
boundaries are themselves `Value` objects.

Example:
    >>> from nasong.core.values.basic.value_random_int import RandomInt
    >>> from nasong.core.values.basic.value_constant import c
    >>> val = RandomInt(min_range=c(1), max_range=c(10))
    >>> 1.0 <= val.get_item(0, 44100) <= 10.0
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
class RandomInt(Value):
    """A Value that returns a random integer within a specified range.

    This class generates stochastic integer values by sampling from a uniform
    distribution and applying a floor operation. The boundaries are defined
    by two boundary Value objects.

    Attributes:
        min_range (Value): The lower bound Value (inclusive).
        max_range (Value): The upper bound Value (inclusive).
    """

    #
    def __init__(self, min_range: Value, max_range: Value) -> None:
        """Initializes the RandomInt Value.

        Args:
            min_range (Value): The Value defining the lower bound.
            max_range (Value): The Value defining the upper bound.
        """
        super().__init__()
        self.min_range: Value = min_range
        self.max_range: Value = max_range

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns a random integer (as float) for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: A random integer within [min, max], returned as a float.
        """
        return float(
            random.randint(
                a=int(self.min_range.get_item(index=index, sample_rate=sample_rate)),
                b=int(self.max_range.get_item(index=index, sample_rate=sample_rate)),
            )
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized array of random integers (as floats).

        This override provides a performance-optimized implementation using
        NumPy's vectorized operations and a floor-based stochastic proxy.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: An array of random integers in float32 format.
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
        ### Vectorized version using numpy's random generator. ###
        #
        min_int: NDArray[np.int64] = min_vals.astype(np.int64)

        #
        ### np.random.randint's 'high' parameter is exclusive, so we add 1 ###
        ### to match the inclusive behavior of random.randint.             ###
        #
        max_int: NDArray[np.int64] = max_vals.astype(np.int64) + 1

        #
        ### Ensure 'high' is always strictly greater than 'low'. ###
        ### If min_int >= max_int, set max_int to min_int + 1.   ###
        #
        max_int = np.maximum(min_int + 1, max_int)

        #
        ### Generate random coefficients for backward pass. ###
        #
        # For RandomInt, the continuous proxy is min + r*(max - min + 1) -> floor()
        random_vals: NDArray[np.float32] = np.random.uniform(
            low=0.0, high=1.0, size=indexes_buffer.shape
        ).astype(np.float32)

        # Save for backward pass
        self._last_random_vals = random_vals

        #
        ### Generate random ints and cast back to float32 for the audio buffer. ###
        #
        return np.floor(min_vals + random_vals * (max_vals - min_vals + 1.0)).astype(
            np.float32
        )

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Returns differentiable random integer values for training.

        The min and max boundaries are treated as differentiable inputs.
        Internally uses a continuous uniform distribution and rounding to
        maintain gradient flow for the boundaries.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of random integers in float32 format.
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
        ### Use continuous uniform distribution (trainable bounds). ###
        ### Round to integers but use smooth random for gradient flow. ###
        #
        random_vals: Tensor = torch.rand_like(
            indexes_buffer, dtype=torch.float32, device=device
        )
        #
        ### Scale to range and round (rounding breaks gradient but bounds are trainable). ###
        #
        return torch.floor(min_vals + random_vals * (max_vals - min_vals + 1.0)).to(
            dtype=torch.float32, device=device
        )

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Straight-through differentiation for RandomInt bounds.

        Uses a continuous proxy to propagate gradients to the boundaries:
        y = floor(min + r*(max - min + 1))
        Proxy dy/dmin = 1 - r
        Proxy dy/dmax = r

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        if hasattr(self, "_last_random_vals"):
            r = self._last_random_vals
            # Straight-through proxy
            self.min_range.backward(grad_output * (1.0 - r), context, sample_rate)
            self.max_range.backward(grad_output * r, context, sample_rate)
