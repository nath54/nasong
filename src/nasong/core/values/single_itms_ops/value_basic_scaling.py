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


#
class BasicScaling(Value):
    """A Value that applies a linear transformation: value * mult_scale + sum_scale."""

    #
    def __init__(self, value: Value, mult_scale: Value, sum_scale: Value) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.mult_scale: Value = mult_scale
        self.sum_scale: Value = sum_scale

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        v: float = self.value.get_item(index=index, sample_rate=sample_rate)
        m: float = self.mult_scale.get_item(index=index, sample_rate=sample_rate)
        s: float = self.sum_scale.get_item(index=index, sample_rate=sample_rate)

        #
        return v * m + s

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        v: NDArray[np.float32] = self.value.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        m: NDArray[np.float32] = self.mult_scale.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        s: NDArray[np.float32] = self.sum_scale.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        return np.multiply(v, m) + s

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        v: Tensor = self.value.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        m: Tensor = self.mult_scale.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        s: Tensor = self.sum_scale.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        return v * m + s

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients through linear scaling.
        y = v * m + s
        dy/dv = m
        dy/dm = v
        dy/ds = 1
        """
        v_val = self.value.getitem_np(context["indices"], sample_rate)
        m_val = self.mult_scale.getitem_np(context["indices"], sample_rate)

        self.value.backward(grad_output * m_val, context, sample_rate)
        self.mult_scale.backward(grad_output * v_val, context, sample_rate)
        self.sum_scale.backward(grad_output, context, sample_rate)
