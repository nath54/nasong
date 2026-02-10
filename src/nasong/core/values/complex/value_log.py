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
import math
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant


#
class Log(Value):
    """A Value that calculates log_base(value)."""

    #
    def __init__(self, value: Value, base: Value = Constant(value=math.e)) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.base: Value = base

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        base_v: float = self.base.get_item(index=index, sample_rate=sample_rate)
        val_v: float = self.value.get_item(index=index, sample_rate=sample_rate)

        #
        return math.log(x=val_v, base=base_v)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        base_v: NDArray[np.float32] = self.base.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        val_v: NDArray[np.float32] = self.value.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        return (np.log(val_v) / np.log(base_v)).astype(dtype=np.float32)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        base_v: Tensor = self.base.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        val_v: Tensor = self.value.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        return (torch.log(val_v) / torch.log(base_v)).to(dtype=torch.float32)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients through log.
        y = ln(x) / ln(b)
        dy/dx = 1 / (x * ln(b))
        dy/db = -ln(x) / (b * (ln(b))^2)
        """
        val_v = self.value.getitem_np(context["indices"], sample_rate)
        base_v = self.base.getitem_np(context["indices"], sample_rate)

        # Avoid log(0) and div by zero
        val_v = np.maximum(1e-7, val_v)
        base_v = np.maximum(1e-7, base_v)

        ln_v = np.log(val_v)
        ln_b = np.log(base_v)

        grad_dx = grad_output / (val_v * ln_b)
        grad_db = -grad_output * ln_v / (base_v * (ln_b**2))

        self.value.backward(grad_dx, context, sample_rate)
        self.base.backward(grad_db, context, sample_rate)
