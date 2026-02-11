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
class HighPass(Value):
    """
    A simple "clipper" Value that limits the minimum value.
    This is NOT an audio filter (like a Butterworth or RC filter).
    It is equivalent to max(value, min_value).
    """

    #
    def __init__(self, value: Value, min_value: Value = Constant(0)) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.min_value: Value = (
            min_value if isinstance(min_value, Value) else Constant(min_value)
        )

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        return max(
            self.min_value.get_item(index=index, sample_rate=sample_rate),
            self.value.get_item(index=index, sample_rate=sample_rate),
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        return np.maximum(
            self.min_value.getitem_np(
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
        return torch.maximum(
            self.min_value.getitem_torch(
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
        Propagate gradients through high-pass (max clipper).
        y = max(x, low)
        dy/dx = 1 if x > low, 0 otherwise.
        dy/dlow = 1 if low >= x, 0 otherwise.
        """
        x_val = self.value.getitem_np(context["indices"], sample_rate)
        low_val = self.min_value.getitem_np(context["indices"], sample_rate)

        mask_x = (x_val > low_val).astype(np.float32)
        mask_low = (low_val >= x_val).astype(np.float32)

        self.value.backward(grad_output * mask_x, context, sample_rate)
        self.min_value.backward(grad_output * mask_low, context, sample_rate)
