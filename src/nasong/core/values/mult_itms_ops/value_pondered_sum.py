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
Pondered sum (weighted sum) implementation.

This module provides the `PonderedSum` class, which calculates a weighted sum
of (weight, value) pairs.

Example:
    >>> from nasong.core.values.basic.value_constant import Constant
    >>> from nasong.core.values.mult_itms_ops.value_pondered_sum import PonderedSum
    >>> w1, v1 = Constant(0.5), Constant(10.0)
    >>> w2, v2 = Constant(0.5), Constant(20.0)
    >>> ps = PonderedSum([(w1, v1), (w2, v2)])
    >>> ps.get_item(0, 44100)
    15.0
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


#
class PonderedSum(Value):
    """A Value that returns a weighted sum of (weight, value) pairs.

    The result is calculated as: (weight1 * value1) + (weight2 * value2) + ...

    Attributes:
        values_and_weights (list[tuple[Value, Value]]): A list of tuples, where
            each tuple contains a weight Value and a target Value.
    """

    #
    def __init__(self, values: list[tuple[Value, Value]]) -> None:
        """Initializes the PonderedSum operation.

        Args:
            values (list[tuple[Value, Value]]): A list of (weight, value) pairs.
        """

        #
        super().__init__()

        #
        self.values_and_weights: list[tuple[Value, Value]] = values

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the weighted sum for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The calculated weighted sum at the given index.
        """

        #
        result: float = 0

        #
        for pond, val in self.values_and_weights:
            #
            result += pond.get_item(
                index=index, sample_rate=sample_rate
            ) * val.get_item(index=index, sample_rate=sample_rate)

        #
        return result

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the weighted sums.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized weighted sum samples.
        """

        #
        result: NDArray[np.float32] = np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        for pond, val in self.values_and_weights:
            #
            result += pond.getitem_np(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate
            ) * val.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        return result

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Generates the weighted sums for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of weighted sum samples.
        """

        #
        if len(self.values_and_weights) == 0:
            #
            return torch.zeros_like(indexes_buffer, dtype=torch.float32, device=device)

        #
        ### Compute weighted sum. ###
        #
        result: Tensor = torch.zeros_like(
            indexes_buffer, dtype=torch.float32, device=device
        )

        #
        for weight, value in self.values_and_weights:
            #
            w_val: Tensor = weight.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            )
            v_val: Tensor = value.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            )
            #
            result = result + w_val * v_val

        #
        return result

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the weighted sum.

        Computes gradients for each weight and each value using the product rule:
        dy/dw_i = x_i
        dy/dx_i = w_i

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        for weight, value in self.values_and_weights:
            w_val = weight.getitem_np(context["indices"], sample_rate)
            v_val = value.getitem_np(context["indices"], sample_rate)

            weight.backward(grad_output * v_val, context, sample_rate)
            value.backward(grad_output * w_val, context, sample_rate)
