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
Naive square wave implementation.

This module provides the `Square` class, which generates a naive square
wave with a variable duty cycle. Note that this waveform contains aliasing;
for audio synthesis, `BandLimitedSquare` is usually preferred.

Example:
    >>> from nasong.core.values.basic.value_identity import Identity
    >>> from nasong.core.values.complex.value_square import Square
    >>> time = Identity()
    >>> sq = Square(value=time, frequency=440.0, duty_cycle=0.5)
    >>> sq.get_item(0, 44100)
    1.0
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
class Square(Value):
    """A Value that generates a "naive" square wave with a variable duty cycle.

    Calculates a signal that is high for a `duty_cycle` portion of the period
    and low for the rest.

    Attributes:
        value (Value): The base phase source.
        frequency (Value): The frequency multiplier.
        amplitude (Value): The peak amplitude.
        delta (Value): The phase offset.
        duty_cycle (Value): The fraction of the period for which the signal is high.
    """

    #
    def __init__(
        self,
        value: Value,
        frequency: Value = Constant(1),
        amplitude: Value = Constant(1),
        delta: Value = Constant(0),
        duty_cycle: Value = Constant(0.5),
    ) -> None:
        """Initializes the Square oscillator.

        Args:
            value (Value): The input phase source (e.g., time).
            frequency (Value): The frequency multiplier (default 1).
            amplitude (Value): The peak amplitude (default 1).
            delta (Value): The phase offset (default 0).
            duty_cycle (Value): The high-state duration fraction (default 0.5).
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
        self.duty_cycle: Value = (
            duty_cycle if isinstance(duty_cycle, Value) else Constant(duty_cycle)
        )

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the square wave value for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The square wave amplitude at the given index.
        """

        #
        val_v: float = self.value.get_item(index=index, sample_rate=sample_rate)
        fre_v: float = self.frequency.get_item(index=index, sample_rate=sample_rate)
        amp_v: float = self.amplitude.get_item(index=index, sample_rate=sample_rate)
        del_v: float = self.delta.get_item(index=index, sample_rate=sample_rate)
        duty_v: float = self.duty_cycle.get_item(index=index, sample_rate=sample_rate)

        #
        ### Calculate the phase and normalize to [0, 1). ###
        #
        phase: float = val_v * fre_v + del_v
        normalized_phase: float = phase - math.floor(phase)

        #
        ### Square wave: high for duty_cycle portion, low for the rest. ###
        #
        if normalized_phase < duty_v:
            #
            return amp_v
        #
        else:
            #
            return -amp_v

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the square wave.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized square wave samples.
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
        duty_v: NDArray[np.float32] = self.duty_cycle.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        ### Calculate the phase and normalize to [0, 1). ###
        #
        phase: NDArray[np.float32] = np.multiply(val_v, fre_v) + del_v
        normalized_phase: NDArray[np.float32] = phase - np.floor(phase)

        #
        ### Square wave: high for duty_cycle portion, low for the rest. ###
        #
        return np.where(normalized_phase < duty_v, amp_v, -amp_v)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Generates the square wave for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of square wave samples.
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
        duty_v: Tensor = self.duty_cycle.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        ### Calculate the phase and normalize to [0, 1). ###
        #
        phase: Tensor = val_v * fre_v + del_v
        normalized_phase: Tensor = phase - torch.floor(phase)

        #
        ### Square wave: high for duty_cycle portion, low for the rest. ###
        #
        return torch.where(normalized_phase < duty_v, amp_v, -amp_v)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the square wave.

        Computes gradients for amplitude. For duty cycle and phase-related inputs,
        it use a straight-through proxy since the naive square wave is not
        differentiable at transitions.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        val_v = self.value.getitem_np(context["indices"], sample_rate)
        f_v = self.frequency.getitem_np(context["indices"], sample_rate)
        amp_v = self.amplitude.getitem_np(context["indices"], sample_rate)
        d_v = self.delta.getitem_np(context["indices"], sample_rate)
        duty = self.duty_cycle.getitem_np(context["indices"], sample_rate)

        phase = val_v * f_v + d_v
        p = phase - np.floor(phase)

        # dy/da
        sq_val = np.where(p < duty, 1.0, -1.0)
        self.amplitude.backward(grad_output * sq_val, context, sample_rate)

        # dy/dduty: straight-through proxy
        # We can approximate dy/dduty = a * pulse at the transition
        # But for now, we leave it at 0 as per the comment.
        self.duty_cycle.backward(np.zeros_like(grad_output), context, sample_rate)

        # Straight-through for phase-related grads
        # Use a sawtooth-like proxy: ramp from -1 to 1 per cycle
        grad_proxy = grad_output * amp_v * 2.0
        self.value.backward(grad_proxy * f_v, context, sample_rate)
        self.frequency.backward(grad_proxy * val_v, context, sample_rate)
        self.delta.backward(grad_proxy, context, sample_rate)
