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
Mask threshold operation implementation.

This module provides the `MaskTreshold` class, which acts as a switch between
two values based on whether a mask value exceeds a threshold.

Example:
    >>> from nasong.core.values.basic.value_constant import Constant
    >>> from nasong.core.values.single_itms_ops.value_mask_treshold import MaskTreshold
    >>> val, mask, threshold, m_val = Constant(0.5), Constant(1.0), Constant(0.8), Constant(0.0)
    >>> mt = MaskTreshold(val, mask, threshold, m_val)
    >>> mt.get_item(0, 44100)
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
class MaskTreshold(Value):
    """A Value that acts as a switch based on a mask and threshold.

    If mask >= threshold, returns mask_value. Otherwise, returns the original value.

    Attributes:
        value (Value): The original value to return if mask < threshold.
        mask (Value): The value to compare against the threshold.
        treshold_to_mask (Value): The threshold value.
        mask_value (Value): The value to return if mask >= threshold.
    """

    #
    def __init__(
        self,
        value: Value,
        mask: Value,
        treshold_to_mask: Value = Constant(1),
        mask_value: Value = Constant(0),
    ) -> None:
        """Initializes the MaskTreshold operation.

        Args:
            value (Value): The original Value object.
            mask (Value): The mask Value object.
            treshold_to_mask (Value, optional): The threshold Value. Defaults to 1.
            mask_value (Value, optional): The value to use when masked. Defaults to 0.
        """

        #
        super().__init__()

        #
        self.value: Value = value
        self.mask: Value = mask
        self.treshold_to_mask: Value = (
            treshold_to_mask
            if isinstance(treshold_to_mask, Value)
            else Constant(treshold_to_mask)
        )
        self.mask_value: Value = (
            mask_value if isinstance(mask_value, Value) else Constant(mask_value)
        )

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the switched value for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: Either mask_value or the original value based on the threshold.
        """

        #
        mask_v: float = self.mask.get_item(index=index, sample_rate=sample_rate)

        #
        if mask_v >= self.treshold_to_mask.get_item(
            index=index, sample_rate=sample_rate
        ):
            #
            return self.mask_value.get_item(index=index, sample_rate=sample_rate)

        #
        else:
            #
            return self.value.get_item(index=index, sample_rate=sample_rate)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the switched values.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized switched samples.
        """

        #
        base_value: NDArray[np.float32] = self.value.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        masked_value: NDArray[np.float32] = self.mask_value.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        mask_v: NDArray[np.float32] = self.mask.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        treshold_v: NDArray[np.float32] = self.treshold_to_mask.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        return np.where(mask_v < treshold_v, base_value, masked_value)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Generates the switched values for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of switched samples.
        """

        #
        base_value: Tensor = self.value.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        masked_value: Tensor = self.mask_value.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        mask_v: Tensor = self.mask.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        treshold_v: Tensor = self.treshold_to_mask.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        return torch.where(mask_v < treshold_v, base_value, masked_value)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the mask threshold operation.

        Gradients are passed to either the original value or the mask value branch
        depending on whether the mask was below or above the threshold.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        mask_v = self.mask.getitem_np(context["indices"], sample_rate)
        tresh_v = self.treshold_to_mask.getitem_np(context["indices"], sample_rate)

        mask_below = (mask_v < tresh_v).astype(np.float32)
        mask_above = (mask_v >= tresh_v).astype(np.float32)

        self.value.backward(grad_output * mask_below, context, sample_rate)
        self.mask_value.backward(grad_output * mask_above, context, sample_rate)
        # Straight-through for mask and threshold
        self.mask.backward(np.zeros_like(grad_output), context, sample_rate)
        self.treshold_to_mask.backward(np.zeros_like(grad_output), context, sample_rate)
