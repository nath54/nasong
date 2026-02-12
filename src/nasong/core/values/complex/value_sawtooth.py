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
Naive sawtooth wave implementation.

This module provides the `Sawtooth` class, which generates a naive sawtooth
wave (linear ramp). Note that this waveform contains aliasing; for audio
synthesis, `BandLimitedSawtooth` is usually preferred.

Example:
    >>> from nasong.core.values.basic.value_identity import Identity
    >>> from nasong.core.values.complex.value_sawtooth import Sawtooth
    >>> time = Identity()
    >>> saw = Sawtooth(value=time, frequency=440.0)
    >>> saw.get_item(0, 44100)
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
class Sawtooth(Value):
    """A Value that generates a "naive" sawtooth wave.

    Calculates a linear ramp from -1 to 1 (rising) or 1 to -1 (falling).

    Attributes:
        value (Value): The base phase source.
        frequency (Value): The frequency multiplier.
        amplitude (Value): The peak amplitude.
        delta (Value): The phase offset.
        direction (Value): Direction of the ramp (>=0 for rising, <0 for falling).
    """

    #
    def __init__(
        self,
        value: Value,
        frequency: Value = Constant(1),
        amplitude: Value = Constant(1),
        delta: Value = Constant(0),
        direction: Value = Constant(1),  # 1 for rising, -1 for falling
    ) -> None:
        """Initializes the Sawtooth oscillator.

        Args:
            value (Value): The input phase source (e.g., time).
            frequency (Value): The frequency multiplier (default 1).
            amplitude (Value): The peak amplitude (default 1).
            delta (Value): The phase offset (default 0).
            direction (Value): Slope direction (default 1, rising).
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
        self.direction: Value = (
            direction if isinstance(direction, Value) else Constant(direction)
        )

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the sawtooth value for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The sawtooth amplitude at the given index.
        """

        #
        val_v: float = self.value.get_item(index=index, sample_rate=sample_rate)
        fre_v: float = self.frequency.get_item(index=index, sample_rate=sample_rate)
        amp_v: float = self.amplitude.get_item(index=index, sample_rate=sample_rate)
        del_v: float = self.delta.get_item(index=index, sample_rate=sample_rate)
        dir_v: float = self.direction.get_item(index=index, sample_rate=sample_rate)

        #
        ### Calculate the phase and normalize to [0, 1). ###
        #
        phase: float = val_v * fre_v + del_v
        normalized_phase: float = phase - math.floor(phase)

        #
        ### Sawtooth wave: linear rise from -1 to 1 (or fall). ###
        #
        if dir_v >= 0:
            #
            ### Rising sawtooth: goes from -1 to 1. ###
            #
            sawtooth_value: float = 2.0 * normalized_phase - 1.0
        #
        else:
            #
            ### Falling sawtooth: goes from 1 to -1. ###
            #
            sawtooth_value: float = 1.0 - 2.0 * normalized_phase

        #
        ### Apply amplitude scaling. ###
        #
        return amp_v * sawtooth_value

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the sawtooth wave.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized sawtooth samples.
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
        dir_v: NDArray[np.float32] = self.direction.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        ### Calculate the phase and normalize to [0, 1). ###
        #
        phase: NDArray[np.float32] = np.multiply(val_v, fre_v) + del_v
        normalized_phase: NDArray[np.float32] = phase - np.floor(phase)

        #
        ### Sawtooth wave: linear rise from -1 to 1 (or fall). ###
        #
        rising_sawtooth: NDArray[np.float32] = (2.0 * normalized_phase - 1.0).astype(
            dtype=np.float32
        )
        falling_sawtooth: NDArray[np.float32] = (1.0 - 2.0 * normalized_phase).astype(
            dtype=np.float32
        )
        sawtooth_value: NDArray[np.float32] = np.where(
            dir_v >= 0, rising_sawtooth, falling_sawtooth
        )

        #
        ### Apply amplitude scaling. ###
        #
        return np.multiply(amp_v, sawtooth_value)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Generates the sawtooth wave for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of sawtooth samples.
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
        dir_v: Tensor = self.direction.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        ### Calculate the phase and normalize to [0, 1). ###
        #
        phase: Tensor = val_v * fre_v + del_v
        normalized_phase: Tensor = phase - torch.floor(phase)

        #
        ### Sawtooth wave: linear rise from -1 to 1 (or fall). ###
        #
        rising_sawtooth: Tensor = (2.0 * normalized_phase - 1.0).to(dtype=torch.float32)
        falling_sawtooth: Tensor = (1.0 - 2.0 * normalized_phase).to(
            dtype=torch.float32
        )
        sawtooth_value: Tensor = torch.where(
            dir_v >= 0, rising_sawtooth, falling_sawtooth
        )

        #
        ### Apply amplitude scaling. ###
        #
        return amp_v * sawtooth_value

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the sawtooth wave.

        Computes gradients for amplitude, base value, frequency, and phase offset.
        Note that the discontinuity (jump) is ignored in the gradient calculation.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        val_v = self.value.getitem_np(context["indices"], sample_rate)
        f_v = self.frequency.getitem_np(context["indices"], sample_rate)
        amp_v = self.amplitude.getitem_np(context["indices"], sample_rate)
        d_v = self.delta.getitem_np(context["indices"], sample_rate)
        dir_v = self.direction.getitem_np(np.zeros_like(grad_output), sample_rate)

        phase = val_v * f_v + d_v
        p = phase - np.floor(phase)

        # dy/da
        saw_val = np.where(dir_v >= 0, 2.0 * p - 1.0, 1.0 - 2.0 * p)
        self.amplitude.backward(grad_output * saw_val, context, sample_rate)

        # dy/dp
        slope = np.where(dir_v >= 0, 2.0, -2.0)
        grad_base = grad_output * amp_v * slope
        self.value.backward(grad_base * f_v, context, sample_rate)
        self.frequency.backward(grad_base * val_v, context, sample_rate)
        self.delta.backward(grad_base, context, sample_rate)
