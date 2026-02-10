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
from typing import Dict, Any
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.input_args import input_args_to_values


#
class Product(Value):
    """A Value that returns the product of a list of input Values."""

    #
    def __init__(self, *values: Value | list[Value]) -> None:

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
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients through product.
        y = prod(x_i)
        dy/dx_i = y / x_i
        """
        y = self.getitem_np(context["indices"], sample_rate)

        for v in self.values:
            xi = v.getitem_np(context["indices"], sample_rate)
            # Avoid division by zero
            grad_xi = grad_output * y / np.where(xi == 0, 1e-7, xi)
            v.backward(grad_xi, context, sample_rate)
