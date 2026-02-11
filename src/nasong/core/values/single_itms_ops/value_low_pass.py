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
from typing import Any
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant


#
class LowPass(Value):
    """
    A simple "clipper" Value that limits the maximum value.
    This is NOT an audio filter (like a Butterworth or RC filter).
    It is equivalent to min(value, max_value).
    """

    #
    def __init__(self, value: Value, max_value: Value = Constant(0)) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.max_value: Value = (
            max_value if isinstance(max_value, Value) else Constant(max_value)
        )

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        return min(
            self.max_value.get_item(index=index, sample_rate=sample_rate),
            self.value.get_item(index=index, sample_rate=sample_rate),
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        return np.minimum(
            self.max_value.getitem_np(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate
            ),
            self.value.getitem_np(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate
            ),
        )

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        return torch.minimum(
            self.max_value.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            ),
            self.value.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            ),
        )

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients through low-pass (min clipper).
        y = min(x, high)
        dy/dx = 1 if x < high, 0 otherwise.
        dy/dhigh = 1 if high <= x, 0 otherwise.
        """
        v_val = self.value.getitem_np(context["indices"], sample_rate)
        high_val = self.max_value.getitem_np(context["indices"], sample_rate)

        mask_x = (v_val < high_val).astype(np.float32)
        mask_high = (high_val <= v_val).astype(np.float32)

        self.value.backward(grad_output * mask_x, context, sample_rate)
        self.max_value.backward(grad_output * mask_high, context, sample_rate)
