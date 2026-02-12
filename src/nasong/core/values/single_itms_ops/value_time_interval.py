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
Time interval operation implementation.

This module provides the `TimeInterval` class, which selects between two
`Value` objects based on whether the current sample index falls within a
specified range.

Example:
    >>> from nasong.core.values.basic.value_constant import Constant
    >>> from nasong.core.values.single_itms_ops.value_time_interval import TimeInterval
    >>> inside, outside = Constant(1.0), Constant(0.0)
    >>> ti = TimeInterval(inside, outside, Constant(100), Constant(200))
    >>> ti.get_item(150, 44100)
    1.0
    >>> ti.get_item(50, 44100)
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
class TimeInterval(Value):
    """A Value that selects between two other Values based on the index (time).

    Returns `value_inside` if `min_sample_idx <= index <= max_sample_idx`.
    Otherwise, returns `value_outside`.

    Attributes:
        value_inside (Value): The value to return when in-range.
        value_outside (Value): The value to return when out-of-range.
        min_sample_idx (Value): The start of the interval.
        max_sample_idx (Value): The end of the interval.
    """

    #
    def __init__(
        self,
        value_inside: Value,
        value_outside: Value = Constant(0),
        min_sample_idx: Value = Constant(0),
        max_sample_idx: Value = Constant(1),
    ) -> None:
        """Initializes the TimeInterval operation.

        Args:
            value_inside (Value): The Value to return inside the interval.
            value_outside (Value, optional): The Value to return outside. Defaults to 0.
            min_sample_idx (Value, optional): The start index. Defaults to 0.
            max_sample_idx (Value, optional): The end index. Defaults to 1.
        """

        #
        super().__init__()

        #
        self.value_inside: Value = value_inside
        self.value_outside: Value = value_outside
        self.min_sample_idx: Value = min_sample_idx
        self.max_sample_idx: Value = max_sample_idx

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the interval-selected value for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: Either the inside value or the outside value.
        """

        #
        if index < self.min_sample_idx.get_item(index=index, sample_rate=sample_rate):
            #
            return self.value_outside.get_item(index=index, sample_rate=sample_rate)

        #
        elif index > self.max_sample_idx.get_item(index=index, sample_rate=sample_rate):
            #
            return self.value_outside.get_item(index=index, sample_rate=sample_rate)

        #
        return self.value_inside.get_item(index=index, sample_rate=sample_rate)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the interval-selected values.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized interval selection samples.
        """

        #
        inside_values: NDArray[np.float32] = self.value_inside.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        outside_values: NDArray[np.float32] = self.value_outside.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        min_idx: NDArray[np.float32] = self.min_sample_idx.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        max_idx: NDArray[np.float32] = self.max_sample_idx.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        ### Create mask for values inside the interval. ###
        #
        inside_mask = (indexes_buffer >= min_idx) & (indexes_buffer <= max_idx)

        #
        return np.where(inside_mask, inside_values, outside_values)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Generates the interval-selected values for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of interval selection samples.
        """

        #
        inside_values: Tensor = self.value_inside.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        outside_values: Tensor = self.value_outside.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        min_idx: Tensor = self.min_sample_idx.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        max_idx: Tensor = self.max_sample_idx.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        ### Create mask for values inside the interval. ###
        #
        inside_mask = (indexes_buffer >= min_idx) & (indexes_buffer <= max_idx)

        #
        return torch.where(inside_mask, inside_values, outside_values)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the time interval operation.

        Gradients are passed back to the `value_inside` or `value_outside` branch
        based on the sample indices.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        # We need the indexes to know which branch was taken
        # grad_output is likely coming from a buffer that has the same shape as the sample indexes
        # But we don't have the indexes directly here.
        # Actually, backward receives grad_output which corresponds to the last getitem_np call.
        # We can reconstruct the mask if we know the sample rate and the size of grad_output.
        # BUT the best is to just call getitem_np on the indices to get the masks.

        # We need the indices. The indices are usually 0..N-1.
        indices = np.arange(len(grad_output), dtype=np.float32)

        min_v = self.min_sample_idx.getitem_np(context["indices"], sample_rate)
        max_v = self.max_sample_idx.getitem_np(context["indices"], sample_rate)

        inside_mask = (indices >= min_v) & (indices <= max_v)
        outside_mask = ~inside_mask

        self.value_inside.backward(
            grad_output * inside_mask.astype(np.float32), context, sample_rate
        )
        self.value_outside.backward(
            grad_output * outside_mask.astype(np.float32), context, sample_rate
        )
        # Straight-through for min/max idx
        self.min_sample_idx.backward(np.zeros_like(grad_output), context, sample_rate)
        self.max_sample_idx.backward(np.zeros_like(grad_output), context, sample_rate)
