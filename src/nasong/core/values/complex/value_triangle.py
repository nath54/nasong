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
Naive triangle wave implementation.

This module provides the `Triangle` class, which generates a naive triangle
wave. Note that this waveform contains aliasing; for audio synthesis,
a band-limited approach is usually preferred.

Example:
    >>> from nasong.core.values.basic.value_identity import Identity
    >>> from nasong.core.values.complex.value_triangle import Triangle
    >>> time = Identity()
    >>> tri = Triangle(value=time, frequency=440.0)
    >>> tri.get_item(0, 44100)
    -1.0
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
class Triangle(Value):
    """A Value that generates a "naive" triangle wave.

    Calculates a linear ramp up and down between -1 and 1.

    Attributes:
        value (Value): The base phase source.
        frequency (Value): The frequency multiplier.
        amplitude (Value): The peak amplitude.
        delta (Value): The phase offset.
    """

    #
    def __init__(
        self,
        value: Value,
        frequency: Value = Constant(1),
        amplitude: Value = Constant(1),
        delta: Value = Constant(0),
    ) -> None:
        """Initializes the Triangle oscillator.

        Args:
            value (Value): The input phase source (e.g., time).
            frequency (Value): The frequency multiplier (default 1).
            amplitude (Value): The peak amplitude (default 1).
            delta (Value): The phase offset (default 0).
        """

        #
        super().__init__()

        #
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
        """Returns the triangle wave value for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The triangle wave amplitude at the given index.
        """

        #
        val_v: float = self.value.get_item(index=index, sample_rate=sample_rate)
        fre_v: float = self.frequency.get_item(index=index, sample_rate=sample_rate)
        amp_v: float = self.amplitude.get_item(index=index, sample_rate=sample_rate)
        del_v: float = self.delta.get_item(index=index, sample_rate=sample_rate)

        #
        ### Calculate the phase. ###
        #
        phase: float = val_v * fre_v + del_v

        #
        ### Triangle wave formula: 2 * |2 * (phase - floor(phase + 0.5))| - 1 ###
        #
        triangle_value: float = 2.0 * abs(2.0 * (phase - math.floor(phase + 0.5))) - 1.0

        #
        ### Apply amplitude scaling. ###
        #
        return amp_v * triangle_value

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the triangle wave.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized triangle wave samples.
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
        ### Calculate the phase. ###
        #
        phase: NDArray[np.float32] = np.multiply(val_v, fre_v) + del_v

        #
        ### Triangle wave formula: 2 * |2 * (phase - floor(phase + 0.5))| - 1 ###
        #
        triangle_value: NDArray[np.float32] = (
            2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0
        ).astype(dtype=np.float32)

        #
        ### Apply amplitude scaling. ###
        #
        return np.multiply(amp_v, triangle_value)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Generates the triangle wave for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of triangle wave samples.
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
        ### Calculate the phase. ###
        #
        phase: Tensor = val_v * fre_v + del_v

        #
        ### Triangle wave formula: 2 * |2 * (phase - floor(phase + 0.5))| - 1 ###
        #
        triangle_value: Tensor = (
            2.0 * torch.abs(2.0 * (phase - torch.floor(phase + 0.5))) - 1.0
        ).to(dtype=torch.float32)

        #
        ### Apply amplitude scaling. ###
        #
        return amp_v * triangle_value

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the triangle wave.

        Computes gradients for amplitude, base value, frequency, and phase offset.
        The slope is handled based on the current phase.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        val_v = self.value.getitem_np(context["indices"], sample_rate)
        f_v = self.frequency.getitem_np(context["indices"], sample_rate)
        amp_v = self.amplitude.getitem_np(context["indices"], sample_rate)
        d_v = self.delta.getitem_np(context["indices"], sample_rate)

        phase = val_v * f_v + d_v
        # Relative phase in [-0.5, 0.5)
        rel_p = phase - np.floor(phase + 0.5)

        # dy/da
        tri_val = 2.0 * np.abs(2.0 * rel_p) - 1.0
        self.amplitude.backward(grad_output * tri_val, context, sample_rate)

        # dy/dp
        # slope is 4*a if rel_p > 0, -4*a if rel_p < 0
        slope_sign = np.where(rel_p >= 0, 1.0, -1.0)
        grad_base = grad_output * amp_v * slope_sign * 4.0
        self.value.backward(grad_base * f_v, context, sample_rate)
        self.frequency.backward(grad_base * val_v, context, sample_rate)
        self.delta.backward(grad_base, context, sample_rate)
