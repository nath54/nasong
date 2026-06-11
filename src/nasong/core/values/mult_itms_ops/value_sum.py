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
Sum operation implementation.

This module provides the `Sum` class, which calculates the sum of a list
of input `Value` objects at each sample.

Example:
    >>> from nasong.core.values.basic.value_constant import Constant
    >>> from nasong.core.values.mult_itms_ops.value_sum import Sum
    >>> v1, v2 = Constant(1.0), Constant(2.0)
    >>> s = Sum(v1, v2)
    >>> s.get_item(0, 44100)
    3.0
"""

#
### Import Modules. ###
#
from typing import Any
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.input_args import input_args_to_values


#
class Sum(Value):
    """A Value that returns the sum of a list of input Values.

    Attributes:
        values (list[Value]): The list of input values to add.
    """

    #
    def __init__(self, *values: Value | list[Value]) -> None:
        """Initializes the Sum operation.

        Args:
            *values (Value | list[Value]): One or more Value objects or a list
                of Value objects to sum.

        Raises:
            ValueError: If any input value is None.
        """

        #
        super().__init__()

        #
        self.values: list[Value] = input_args_to_values(values=values)
        for i, v in enumerate(self.values):
            if v is None:
                raise ValueError(
                    f"Sum created with None value at index {i}. input_args_to_values returned {self.values}"
                )

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the sum for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The sum of all input amplitudes at the given index.
        """

        #
        return sum(
            [v.get_item(index=index, sample_rate=sample_rate) for v in self.values]
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the sums.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized sum samples.
        """
        if len(self.values) == 0:
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        total = self.values[0].getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        for v in self.values[1:]:
            total = total + v.getitem_np(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate
            )
        return total

    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Generates the sums for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of sum samples.
        """
        if len(self.values) == 0:
            return torch.zeros_like(indexes_buffer, dtype=torch.float32, device=device)

        total = self.values[0].getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        for val in self.values[1:]:
            total = total + val.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            )
        return total

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the sum operation.

        Uses the linearity of addition: dy/dx_i = 1.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        for v in self.values:
            v.backward(grad_output, context, sample_rate)
