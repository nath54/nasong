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
Product operation implementation.

This module provides the `Product` class, which calculates the product of a list
of input `Value` objects at each sample.

Example:
    >>> from nasong.core.values.basic.value_constant import Constant
    >>> from nasong.core.values.mult_itms_ops.value_product import Product
    >>> v1, v2 = Constant(2.0), Constant(3.0)
    >>> p = Product(v1, v2)
    >>> p.get_item(0, 44100)
    6.0
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
class Product(Value):
    """A Value that returns the product of a list of input Values.

    Attributes:
        values (list[Value]): The list of input values to multiply.
    """

    #
    def __init__(self, *values: Value | list[Value]) -> None:
        """Initializes the Product operation.

        Args:
            *values (Value | list[Value]): One or more Value objects or a list
                of Value objects to multiply.

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
                    f"Product created with None value at index {i}. input_args_to_values returned {self.values}"
                )

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the product for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The product of all input amplitudes at the given index.
        """

        #
        result: float = 1

        #
        for v in self.values:
            #
            result *= v.get_item(index=index, sample_rate=sample_rate)

        #
        return result

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the products.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized product samples.
        """

        #
        result: NDArray[np.float32] = np.ones_like(indexes_buffer, dtype=np.float32)

        #
        for v in self.values:
            #
            result = np.multiply(
                result,
                v.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate),
            )

        #
        return result

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Generates the products for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of product samples.
        """

        #
        result: Tensor = torch.ones_like(
            indexes_buffer, dtype=torch.float32, device=device
        )

        #
        for v in self.values:
            #
            result = result * v.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            )

        #
        return result

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the product operation.

        Uses the product rule: dy/dx_i = y / x_i (approximate for efficiency).

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        y = self.getitem_np(context["indices"], sample_rate)

        for v in self.values:
            xi = v.getitem_np(context["indices"], sample_rate)
            # Avoid division by zero
            grad_xi = grad_output * y / np.where(xi == 0, 1e-7, xi)
            v.backward(grad_xi, context, sample_rate)
