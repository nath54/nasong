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
class Sum(Value):
    """A Value that returns the sum of a list of input Values."""

    #
    def __init__(self, *values: Value | list[Value]) -> None:

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

        #
        return sum(
            [v.get_item(index=index, sample_rate=sample_rate) for v in self.values]
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        arrays = [
            v.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
            for v in self.values
        ]

        # Autograd fix: np.sum(list) is not supported, use stack.
        stacked = np.stack(arrays, axis=0)
        return np.sum(stacked, axis=0)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

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
            return torch.sum(stacked, dim=0)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients through sum.
        y = sum(x_i)
        dy/dx_i = 1
        """
        for v in self.values:
            v.backward(grad_output, context, sample_rate)
