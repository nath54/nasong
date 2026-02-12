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
Guitar distortion effect implementation.

This module provides the `Distortion` class, which implements a soft-clipping
distortion effect using the hyperbolic tangent (`tanh`) function.

Example:
    >>> from nasong.core.values.basic.value_constant import Constant
    >>> from nasong.core.values.complex.value_distortion import Distortion
    >>> signal = Constant(0.5)
    >>> dist = Distortion(value=signal, gain=10.0)
    >>> dist.get_item(0, 44100)
    0.4999...
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
class Distortion(Value):
    """Guitar distortion effect using `tanh` soft clipping.

    This class applies gain to an input signal and then clips it using the
    `tanh` function to create a warm, analog-style distortion.

    Attributes:
        value (Value): The input audio signal.
        gain (Value): The distortion gain (higher = more clipping).
    """

    #
    def __init__(self, value: Value, gain: Value = Constant(5.0)) -> None:
        """Initializes the Distortion effect.

        Args:
            value (Value): The input audio signal to be distorted.
            gain (Value): The amount of gain before clipping.
        """

        #
        super().__init__()

        #
        self.value: Value = value
        self.gain: Value = gain

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the distorted sample value for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The distorted signal amplitude at the given index.
        """

        #
        ### Apply gain. ###
        #
        x: float = self.value.get_item(
            index=index, sample_rate=sample_rate
        ) * self.gain.get_item(index=index, sample_rate=sample_rate)

        #
        ### Soft clipping using tanh. ###
        #
        return math.tanh(x) * 0.5

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the distorted signal.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized distorted signal samples.
        """

        #
        ### Get the input signal buffer and apply gain. ###
        #
        x: NDArray[np.float32] = self.value.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        ) * self.gain.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        ### Apply vectorized soft clipping using np.tanh. ###
        #
        return (np.tanh(x) * 0.5).astype(dtype=np.float32)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Generates the distorted signal for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of distorted signal samples.
        """

        #
        x: Tensor = self.value.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        gain_v: Tensor = self.gain.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        ### Amplify the signal. ###
        #
        x = x * gain_v

        #
        ### Apply vectorized soft clipping using torch.tanh. ###
        #
        return (torch.tanh(x) * 0.5).to(dtype=torch.float32)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the tanh distortion.

        Computes gradients for the input signal and gain using the derivative
        of tanh: d/du tanh(u) = 1 - tanh^2(u).

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        val_v = self.value.getitem_np(np.zeros_like(grad_output), sample_rate)
        gain_v = self.gain.getitem_np(np.zeros_like(grad_output), sample_rate)

        tx = val_v * gain_v
        tanh_tx = np.tanh(tx)

        # d/du tanh(u) = 1 - tanh^2(u)
        grad_base = grad_output * 0.5 * (1.0 - tanh_tx**2)

        self.value.backward(grad_base * gain_v, context, sample_rate)
        self.gain.backward(grad_base * val_v, context, sample_rate)
