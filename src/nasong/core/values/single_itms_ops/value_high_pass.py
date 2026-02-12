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
High-pass (minimum clipper) operation implementation.

This module provides the `HighPass` class, which limits the minimum amplitude
of an input `Value` object. Note that this is a simple mathematical clipper,
not a frequency-domain filter.

Example:
    >>> from nasong.core.values.basic.value_constant import Constant
    >>> from nasong.core.values.single_itms_ops.value_high_pass import HighPass
    >>> v, mi = Constant(-1.0), Constant(0.0)
    >>> hp = HighPass(v, mi)
    >>> hp.get_item(0, 44100)
    0.0
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
    """A simple "clipper" Value that limits the minimum value.

    This is NOT an audio filter (like a Butterworth or RC filter).
    It is equivalent to max(value, min_value).

    Attributes:
        value (Value): The source value to clip.
        min_value (Value): The lower bound.
    """

    #
    def __init__(self, value: Value, min_value: Value = Constant(0)) -> None:
        """Initializes the HighPass operation.

        Args:
            value (Value): The input Value object.
            min_value (Value, optional): The minimum allowed value. Defaults to 0.
        """

        #
        super().__init__()

        #
        self.value: Value = value
        self.min_value: Value = (
            min_value if isinstance(min_value, Value) else Constant(min_value)
        )

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the high-passed (clipped) value for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The amplitude constrained to the minimum value.
        """

        #
        return max(
            self.min_value.get_item(index=index, sample_rate=sample_rate),
            self.value.get_item(index=index, sample_rate=sample_rate),
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the high-passed values.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized high-passed samples.
        """

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
        """Generates the high-passed values for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of high-passed samples.
        """

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
        """Propagates gradients through the high-pass operation.

        Uses a mask where x > low: dy/dx = 1, dy/dlow = 0.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        x_val = self.value.getitem_np(context["indices"], sample_rate)
        low_val = self.min_value.getitem_np(context["indices"], sample_rate)

        mask_x = (x_val > low_val).astype(np.float32)
        mask_low = (low_val >= x_val).astype(np.float32)

        self.value.backward(grad_output * mask_x, context, sample_rate)
        self.min_value.backward(grad_output * mask_low, context, sample_rate)
