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
Clamping (clipping) operation implementation.

This module provides the `Clamp` class, which constrains the values of an
input `Value` object between a specified minimum and maximum.

Example:
    >>> from nasong.core.values.basic.value_constant import Constant
    >>> from nasong.core.values.single_itms_ops.value_clamp import Clamp
    >>> v, mi, ma = Constant(1.5), Constant(0.0), Constant(1.0)
    >>> cl = Clamp(v, mi, ma)
    >>> cl.get_item(0, 44100)
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
class Clamp(Value):
    """A Value that constrains another Value between a min and max Value.

    Also known as Clip. Constrains all samples to the range [min_value, max_value].

    Attributes:
        value (Value): The source value to clamp.
        min_value (Value): The lower bound.
        max_value (Value): The upper bound.
    """

    #
    def __init__(
        self,
        value: Value,
        min_value: Value = Constant(0),
        max_value: Value = Constant(1),
    ) -> None:
        """Initializes the Clamp operation.

        Args:
            value (Value): The input Value object.
            min_value (Value, optional): The minimum allowed value. Defaults to 0.
            max_value (Value, optional): The maximum allowed value. Defaults to 1.
        """

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
        """Returns the clamped value for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The clamped amplitude at the given index.
        """

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
        """Returns a vectorized NumPy array of the clamped values.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized clamped samples.
        """

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
        """Generates the clamped values for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of clamped samples.
        """

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
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the clamp operation.

        Gradients are passed back to the input, min_value, or max_value depending
        on which branch was active at each sample.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
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
