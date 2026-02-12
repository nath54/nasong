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
Low-pass (maximum clipper) operation implementation.

This module provides the `LowPass` class, which limits the maximum amplitude
of an input `Value` object. Note that this is a simple mathematical clipper,
not a frequency-domain filter.

Example:
    >>> from nasong.core.values.basic.value_constant import Constant
    >>> from nasong.core.values.single_itms_ops.value_low_pass import LowPass
    >>> v, ma = Constant(1.5), Constant(1.0)
    >>> lp = LowPass(v, ma)
    >>> lp.get_item(0, 44100)
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
from nasong.core.values.basic.value_constant import Constant


#
class LowPass(Value):
    """A simple "clipper" Value that limits the maximum value.

    This is NOT an audio filter (like a Butterworth or RC filter).
    It is equivalent to min(value, max_value).

    Attributes:
        value (Value): The source value to clip.
        max_value (Value): The upper bound.
    """

    #
    def __init__(self, value: Value, max_value: Value = Constant(0)) -> None:
        """Initializes the LowPass operation.

        Args:
            value (Value): The input Value object.
            max_value (Value, optional): The maximum allowed value. Defaults to 0.
        """

        #
        super().__init__()

        #
        self.value: Value = value
        self.max_value: Value = (
            max_value if isinstance(max_value, Value) else Constant(max_value)
        )

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the low-passed (clipped) value for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The amplitude constrained to the maximum value.
        """

        #
        return min(
            self.max_value.get_item(index=index, sample_rate=sample_rate),
            self.value.get_item(index=index, sample_rate=sample_rate),
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the low-passed values.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized low-passed samples.
        """

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
        """Generates the low-passed values for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of low-passed samples.
        """

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
        """Propagates gradients through the low-pass operation.

        Uses a mask where x < high: dy/dx = 1, dy/dhigh = 0.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        v_val = self.value.getitem_np(context["indices"], sample_rate)
        high_val = self.max_value.getitem_np(context["indices"], sample_rate)

        mask_x = (v_val < high_val).astype(np.float32)
        mask_high = (high_val <= v_val).astype(np.float32)

        self.value.backward(grad_output * mask_x, context, sample_rate)
        self.max_value.backward(grad_output * mask_high, context, sample_rate)
