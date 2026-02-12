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
Identity Value implementation.

This module provides the `Identity` class, which represents a Value that returns
the sample index itself as the value. This is typically used as the base
representation of time in the audio generation process.

Example:
    >>> from nasong.core.values.basic.value_identity import Identity
    >>> val = Identity()
    >>> val.get_item(100, 44100)
    100.0
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
class Identity(Value):
    """A Value that returns the sample index itself as the value.

    This class serves as the identity function for audio sample indexes,
    effectively representing the raw time-step index.
    """

    #
    def __init__(self) -> None:
        """Initializes the Identity Value."""
        super().__init__()

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the sample index as a float.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate (ignored).

        Returns:
            float: The index value.
        """
        return float(index)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns the indexes buffer itself.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate (ignored).

        Returns:
            NDArray[np.float32]: The original indexes_buffer.
        """
        return indexes_buffer

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Returns the indexes buffer moved to the specified device.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate (ignored).
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: The indexes_buffer on the specified device.
        """
        return indexes_buffer.to(device)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates the gradient.

        Identity is the time input, so the backward pass does nothing.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        pass
