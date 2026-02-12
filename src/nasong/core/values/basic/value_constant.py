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
Constant Value implementation.

This module provides the `Constant` class, which represents a Value that returns
a fixed constant number for all indexes. It also provides a shorthand helper
function `c` for easy instantiation.

Example:
    >>> from nasong.core.values.basic.value_constant import Constant, c
    >>> val = Constant(440.0)
    >>> val.get_item(0, 44100)
    440.0
    >>> val2 = c(100)
    >>> val2.get_item(100, 44100)
    100.0
"""

#
### Import Modules. ###
#
import numpy as np
from numpy.typing import NDArray

from typing import Any

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor


#
class Constant(Value):
    """A Value that returns the same constant number for all indexes.

    This class provides a constant value that does not change over time
    (across different sample indexes).

    Attributes:
        value (float | int): The constant number to return.
    """

    #
    def __init__(self, value: float | int) -> None:
        """Initializes the Constant Value.

        Args:
            value (float | int): The constant value to return for all samples.
        """
        super().__init__()
        self.value: float | int = value

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the constant value for a specific index.

        Args:
            index (int): The sample index (ignored).
            sample_rate (int): The audio sample rate (ignored).

        Returns:
            float: The constant value.
        """
        return float(self.value)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a NumPy array filled with the constant value.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate (ignored).

        Returns:
            NDArray[np.float32]: An array of the same shape as indexes_buffer,
                filled with the constant value.
        """
        return np.full_like(
            indexes_buffer, fill_value=float(self.value), dtype=np.float32
        )

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Returns a PyTorch tensor filled with the constant value.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate (ignored).
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of the same shape as indexes_buffer,
                filled with the constant value.
        """
        return torch.full_like(
            indexes_buffer,
            fill_value=float(self.value),
            dtype=torch.float32,
            device=device,
        )

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates the gradient.

        Constant has no inputs, so the backward pass does nothing.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        pass


#
def c(v: float | int) -> Value:
    """Shorthand helper function to create a Constant Value.

    Args:
        v (float | int): The constant value.

    Returns:
        Value: A new Constant Value instance.
    """
    return Constant(value=v)
