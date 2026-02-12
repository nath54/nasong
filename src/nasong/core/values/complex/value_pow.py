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
Power function implementation.

This module provides the `Pow` class, which calculates the power of a base raised
to an exponent.

Example:
    >>> from nasong.core.values.basic.value_constant import Constant
    >>> from nasong.core.values.complex.value_pow import Pow
    >>> base = Constant(2.0)
    >>> exponent = Constant(3.0)
    >>> p = Pow(base=base, exponent=exponent)
    >>> p.get_item(0, 44100)
    8.0
"""

#
### Import Modules. ###
#
from typing import Any
import math
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant


#
class Pow(Value):
    """A Value that calculates base ^ exponent.

    Attributes:
        exponent (Value): The exponent value.
        base (Value): The base value (default is e).
    """

    #
    def __init__(self, exponent: Value, base: Value = Constant(value=math.e)) -> None:
        """Initializes the Pow value.

        Args:
            exponent (Value): The exponent.
            base (Value): The base (default math.e).
        """

        #
        super().__init__()

        #
        self.exponent: Value = exponent
        self.base: Value = base

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the power value for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The calculated base ^ exponent at the given index.
        """

        #
        base_v: float = self.base.get_item(index=index, sample_rate=sample_rate)
        exp_v: float = self.exponent.get_item(index=index, sample_rate=sample_rate)

        #
        return base_v**exp_v

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the power samples.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized power samples.
        """

        #
        base_v: NDArray[np.float32] = self.base.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        exp_v: NDArray[np.float32] = self.exponent.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        return np.power(base_v, exp_v)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Generates the power samples for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of power samples.
        """

        #
        base_v: Tensor = self.base.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        exp_v: Tensor = self.exponent.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        return torch.pow(base_v, exp_v)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the power function.

        Computes gradients for the base and exponent using the chain rule:
        dy/db = e * b^(e-1)
        dy/de = b^e * ln(b)

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        b_val = self.base.getitem_np(context["indices"], sample_rate)
        e_val = self.exponent.getitem_np(context["indices"], sample_rate)

        # dy/db
        grad_db = grad_output * e_val * np.power(np.maximum(1e-7, b_val), e_val - 1.0)

        # dy/de
        y = np.power(b_val, e_val)
        grad_de = grad_output * y * np.log(np.maximum(1e-7, b_val))

        self.base.backward(grad_db, context, sample_rate)
        self.exponent.backward(grad_de, context, sample_rate)
