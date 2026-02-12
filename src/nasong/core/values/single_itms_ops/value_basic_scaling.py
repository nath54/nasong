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
Linear scaling operation implementation.

This module provides the `BasicScaling` class, which calculates a linear
transformation of an input `Value` object.

Example:
    >>> from nasong.core.values.basic.value_constant import Constant
    >>> from nasong.core.values.single_itms_ops.value_basic_scaling import BasicScaling
    >>> v, m, s = Constant(1.0), Constant(2.0), Constant(0.5)
    >>> sc = BasicScaling(v, m, s)
    >>> sc.get_item(0, 44100)
    2.5
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
    """A Value that applies a linear transformation: value * mult_scale + sum_scale.

    Attributes:
        value (Value): The source value to scale.
        mult_scale (Value): The multiplier (gain).
        sum_scale (Value): The offset (bias).
    """

    #
    def __init__(self, value: Value, mult_scale: Value, sum_scale: Value) -> None:
        """Initializes the BasicScaling operation.

        Args:
            value (Value): The source Value.
            mult_scale (Value): The gain Value.
            sum_scale (Value): The bias Value.
        """

        #
        super().__init__()

        #
        self.value: Value = value
        self.mult_scale: Value = mult_scale
        self.sum_scale: Value = sum_scale

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the scaled value for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The calculated linear transformation (v * m + s).
        """

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
        """Returns a vectorized NumPy array of the scaled values.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized scaled samples.
        """

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
        """Generates the scaled values for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of scaled samples.
        """

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
        """Propagates gradients through the linear scaling operation.

        Computes gradients for the value, multiplier, and offset using the
        product and sum rules.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        v_val = self.value.getitem_np(context["indices"], sample_rate)
        m_val = self.mult_scale.getitem_np(context["indices"], sample_rate)

        self.value.backward(grad_output * m_val, context, sample_rate)
        self.mult_scale.backward(grad_output * v_val, context, sample_rate)
        self.sum_scale.backward(grad_output, context, sample_rate)
