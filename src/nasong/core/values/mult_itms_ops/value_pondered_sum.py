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


#
class PonderedSum(Value):
    """
    A Value that returns a weighted sum of (weight, value) pairs.
    Result = (weight1 * value1) + (weight2 * value2) + ...
    """

    #
    def __init__(self, values: list[tuple[Value, Value]]) -> None:

        #
        super().__init__()

        #
        self.values_and_weights: list[tuple[Value, Value]] = values

    #
    def get_item(self, index: int, sample_rate: int) -> float:

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
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients through pondered sum.
        y = sum(w_i * x_i)
        dy/dw_i = x_i
        dy/dx_i = w_i
        """
        for weight, value in self.values_and_weights:
            w_val = weight.getitem_np(context["indices"], sample_rate)
            v_val = value.getitem_np(context["indices"], sample_rate)

            weight.backward(grad_output * v_val, context, sample_rate)
            value.backward(grad_output * w_val, context, sample_rate)
