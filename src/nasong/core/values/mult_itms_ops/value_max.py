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
Maximum operation implementation.

This module provides the `Max` class, which returns the maximum value among
a list of input `Value` objects at each sample.

Example:
    >>> from nasong.core.values.basic.value_constant import Constant
    >>> from nasong.core.values.mult_itms_ops.value_max import Max
    >>> v1 = Constant(1.0)
    >>> v2 = Constant(2.0)
    >>> m = Max(v1, v2)
    >>> m.get_item(0, 44100)
    2.0
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
class Max(Value):
    """A Value that returns the maximum value from a list of input Values.

    Attributes:
        values (list[Value]): The list of input values to compare.
    """

    #
    def __init__(self, *values: Value | list[Value]) -> None:
        """Initializes the Max operation.

        Args:
            *values (Value | list[Value]): One or more Value objects or a list
                of Value objects to find the maximum of.
        """

        #
        super().__init__()

        #
        self.values: list[Value] = input_args_to_values(values=values)

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the maximum value for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The maximum amplitude among all inputs at the given index.
        """

        #
        return max(
            [v.get_item(index=index, sample_rate=sample_rate) for v in self.values]
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the maximum values.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized maximum samples.
        """

        #
        arrays = [
            v.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
            for v in self.values
        ]

        # Autograd fix: np.maximum.reduce is not supported, use stack + max.
        stacked = np.stack(arrays, axis=0)
        return np.max(stacked, axis=0)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Generates the maximum values for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of maximum samples.
        """

        #
        ### Compute all values and stack them. ###
        #
        value_tensors: list[Tensor] = [
            val.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            )
            for val in self.values
        ]

        #
        if len(value_tensors) == 0:
            #
            return torch.zeros_like(indexes_buffer, dtype=torch.float32, device=device)
        #
        elif len(value_tensors) == 1:
            #
            return value_tensors[0]
        #
        else:
            #
            stacked: Tensor = torch.stack(value_tensors, dim=0)
            #
            return torch.max(stacked, dim=0)[0]

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the max operation.

        Gradients are passed only to the input that provided the maximum value
        at each sample (using a mask).

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        if not self.values:
            return

        arrays = [
            v.getitem_np(np.zeros_like(grad_output), sample_rate) for v in self.values
        ]
        stacked = np.stack(arrays, axis=0)
        max_idx = np.argmax(stacked, axis=0)

        for i, v in enumerate(self.values):
            mask = (max_idx == i).astype(np.float32)
            v.backward(grad_output * mask, context, sample_rate)
