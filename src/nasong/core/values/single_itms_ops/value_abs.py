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
class Abs(Value):
    """A Value that returns the absolute value of another Value."""

    #
    def __init__(self, value: Value) -> None:

        #
        super().__init__()

        #
        self.value: Value = value

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        return abs(self.value.get_item(index=index, sample_rate=sample_rate))

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        return np.abs(
            self.value.getitem_np(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate
            )
        )

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        return torch.abs(
            self.value.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            )
        )

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients through abs.
        y = |x|
        dy/dx = sign(x)
        """
        x_v = self.value.getitem_np(context["indices"], sample_rate)
        self.value.backward(grad_output * np.sign(x_v), context, sample_rate)
