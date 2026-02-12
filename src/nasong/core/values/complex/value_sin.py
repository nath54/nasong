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
Sine wave implementation.

This module provides the `Sin` class, which generates a sine wave based on
an input value, frequency, amplitude, and phase offset (delta).

Example:
    >>> from nasong.core.values.basic.value_identity import Identity
    >>> from nasong.core.values.complex.value_sin import Sin
    >>> time = Identity()
    >>> sin_wave = Sin(value=time, frequency=440.0)
    >>> sin_wave.get_item(0, 44100)
    0.0
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
class Sin(Value):
    """A Value that generates a sine wave.

    Calculates: amplitude * sin( (value * frequency) + delta )

    Attributes:
        value (Value): The base value (often time or phase).
        frequency (Value): The frequency multiplier.
        amplitude (Value): The peak amplitude.
        delta (Value): The phase offset in radians.
    """

    #
    def __init__(
        self,
        value: Value,
        frequency: Value = Constant(1),
        amplitude: Value = Constant(1),
        delta: Value = Constant(0),
    ) -> None:
        """Initializes the Sin wave.

        Args:
            value (Value): The base input value.
            frequency (Value): The frequency multiplier (default 1).
            amplitude (Value): The peak amplitude (default 1).
            delta (Value): The phase offset (default 0).
        """

        #
        super().__init__()

        #
        # Ensure all inputs are Value objects
        self.value: Value = value
        self.frequency: Value = (
            frequency if isinstance(frequency, Value) else Constant(frequency)
        )
        self.amplitude: Value = (
            amplitude if isinstance(amplitude, Value) else Constant(amplitude)
        )
        self.delta: Value = delta if isinstance(delta, Value) else Constant(delta)

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the sine value for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The sine amplitude at the given index.
        """

        #
        val_v: float = self.value.get_item(index=index, sample_rate=sample_rate)
        fre_v: float = self.frequency.get_item(index=index, sample_rate=sample_rate)
        amp_v: float = self.amplitude.get_item(index=index, sample_rate=sample_rate)
        del_v: float = self.delta.get_item(index=index, sample_rate=sample_rate)

        #
        return amp_v * math.sin(val_v * fre_v + del_v)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the sine wave.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized wave samples.
        """

        #
        val_v: NDArray[np.float32] = self.value.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        fre_v: NDArray[np.float32] = self.frequency.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        amp_v: NDArray[np.float32] = self.amplitude.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        del_v: NDArray[np.float32] = self.delta.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        return np.multiply(amp_v, np.sin(np.multiply(val_v, fre_v) + del_v))

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Generates the sine wave for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of wave samples.
        """

        #
        val_v: Tensor = self.value.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        fre_v: Tensor = self.frequency.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        amp_v: Tensor = self.amplitude.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        del_v: Tensor = self.delta.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        return amp_v * torch.sin(val_v * fre_v + del_v)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the sine wave.

        Computes gradients for amplitude, base value, frequency, and phase (delta)
        using the chain rule.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        val_v = self.value.getitem_np(context["indices"], sample_rate)
        fre_v = self.frequency.getitem_np(context["indices"], sample_rate)
        amp_v = self.amplitude.getitem_np(context["indices"], sample_rate)
        del_v = self.delta.getitem_np(context["indices"], sample_rate)

        phase = val_v * fre_v + del_v
        sin_phase = np.sin(phase)
        cos_phase = np.cos(phase)

        self.amplitude.backward(grad_output * sin_phase, context, sample_rate)

        grad_base = grad_output * amp_v * cos_phase
        self.value.backward(grad_base * fre_v, context, sample_rate)
        self.frequency.backward(grad_base * val_v, context, sample_rate)
        self.delta.backward(grad_base, context, sample_rate)
