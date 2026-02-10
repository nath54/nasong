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
from nasong.core.values.basic.value_constant import Constant


#
class Clamp(Value):
    """
    A Value that constrains another Value between a min and max Value.
    Also known as Clip.
    """

    #
    def __init__(
        self,
        value: Value,
        min_value: Value = Constant(0),
        max_value: Value = Constant(1),
    ) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.min_value: Value = (
            min_value if isinstance(min_value, Value) else Constant(min_value)
        )
        self.max_value: Value = (
            max_value if isinstance(max_value, Value) else Constant(max_value)
        )

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        return max(
            self.min_value.get_item(index=index, sample_rate=sample_rate),
            min(
                self.max_value.get_item(index=index, sample_rate=sample_rate),
                self.value.get_item(index=index, sample_rate=sample_rate),
            ),
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        return np.clip(
            self.value.getitem_np(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate
            ),
            self.min_value.getitem_np(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate
            ),
            self.max_value.getitem_np(
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
        return torch.clamp(
            self.value.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            ),
            self.min_value.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            ),
            self.max_value.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            ),
        )

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients through clamp.
        y = clip(x, low, high)
        dy/dx = 1 if low < x < high, 0 otherwise.
        dy/dlow = 1 if x < low, 0 otherwise.
        dy/dhigh = 1 if x > high, 0 otherwise.
        """
        x_val = self.value.getitem_np(context["indices"], sample_rate)
        low_val = self.min_value.getitem_np(context["indices"], sample_rate)
        high_val = self.max_value.getitem_np(context["indices"], sample_rate)

        mask_x = (x_val > low_val) & (x_val < high_val)
        mask_low = x_val <= low_val
        mask_high = x_val >= high_val

        self.value.backward(
            grad_output * mask_x.astype(np.float32), context, sample_rate
        )
        self.min_value.backward(
            grad_output * mask_low.astype(np.float32), context, sample_rate
        )
        self.max_value.backward(
            grad_output * mask_high.astype(np.float32), context, sample_rate
        )
