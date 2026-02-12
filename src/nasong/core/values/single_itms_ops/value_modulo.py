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
Modulo (remainder) operation implementation.

This module provides the `Modulo` class, which computes the remainder of a
division between two `Value` objects.

Example:
    >>> from nasong.core.values.basic.value_constant import Constant
    >>> from nasong.core.values.single_itms_ops.value_modulo import Modulo
    >>> v, m = Constant(5.5), Constant(1.0)
    >>> mo = Modulo(v, m)
    >>> mo.get_item(0, 44100)
    0.5
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
class Modulo(Value):
    """A Value that computes the modulo (remainder) of another Value.

    Result = value % modulo_value. Ideal for creating looping LFOs.

    Attributes:
        value (Value): The dividend Value.
        modulo_value (Value): The divisor Value.
    """

    #
    def __init__(self, value: Value, modulo_value: Value = Constant(1.0)) -> None:
        """Initializes the Modulo operation.

        Args:
            value (Value): The input Value object.
            modulo_value (Value, optional): The modulo divisor. Defaults to 1.0.
        """
        super().__init__()
        self.value: Value = value
        self.modulo_value: Value = (
            modulo_value if isinstance(modulo_value, Value) else Constant(modulo_value)
        )

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the modulo for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The remainder of the division. Returns the value itself if
                the divisor is zero.
        """

        #
        val_v: float = self.value.get_item(index=index, sample_rate=sample_rate)
        mod_v: float = self.modulo_value.get_item(index=index, sample_rate=sample_rate)

        #
        ### Handle division by zero. ###
        #
        if mod_v == 0:
            #
            return val_v

        #
        ### Use % operator for correct "wrapping" behavior. ###
        #
        return val_v % mod_v

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the modulo values.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized modulo samples.
        """

        #
        val_v: NDArray[np.float32] = self.value.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        mod_v: NDArray[np.float32] = self.modulo_value.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        ### Use np.mod for vectorized modulo. ###
        ### We use np.where to prevent division by zero. ###
        #
        return np.where(mod_v == 0, val_v, np.mod(val_v, mod_v))

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Generates the modulo values for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of modulo samples.
        """

        #
        val_v: Tensor = self.value.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        mod_v: Tensor = self.modulo_value.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        ### Use torch.fmod for vectorized modulo. ###
        ### We use torch.where to prevent division by zero. ###
        #
        return torch.where(mod_v == 0, val_v, torch.fmod(val_v, mod_v))

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the modulo operation.

        Uses the identity for the input (ignoring discontinuities) and zero for
        the divisor.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        self.value.backward(grad_output, context, sample_rate)
        self.modulo_value.backward(np.zeros_like(grad_output), context, sample_rate)
