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
Absolute value operation implementation.

This module provides the `Abs` class, which calculates the absolute value
of an input `Value` object at each sample.

Example:
    >>> from nasong.core.values.basic.value_constant import Constant
    >>> from nasong.core.values.single_itms_ops.value_abs import Abs
    >>> v = Constant(-1.0)
    >>> a = Abs(v)
    >>> a.get_item(0, 44100)
    1.0
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
class Abs(Value):
    """A Value that returns the absolute value of another Value.

    Attributes:
        value (Value): The input value to transform.
    """

    #
    def __init__(self, value: Value) -> None:
        """Initializes the Abs operation.

        Args:
            value (Value): The input Value object.
        """

        #
        super().__init__()

        #
        self.value: Value = value

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the absolute value for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The absolute value of the input amplitude.
        """

        #
        return abs(self.value.get_item(index=index, sample_rate=sample_rate))

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the absolute values.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized absolute samples.
        """

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
        """Generates the absolute values for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of absolute samples.
        """

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
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the absolute value operation.

        Uses the sign of the input: dy/dx = sign(x).

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        x_v = self.value.getitem_np(context["indices"], sample_rate)
        self.value.backward(grad_output * np.sign(x_v), context, sample_rate)
