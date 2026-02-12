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
Band-limited Sawtooth wave implementation.

This module provides the `BandLimitedSawtooth` class, which generates a
sawtooth wave by summing a finite number of sine wave harmonics. This
approach avoids the aliasing issues of a "naive" sawtooth wave.

Example:
    >>> from nasong.core.values.basic.value_identity import Identity
    >>> from nasong.core.values.complex.value_band_limited_sawtooth import BandLimitedSawtooth
    >>> time = Identity()
    >>> saw = BandLimitedSawtooth(time=time, frequency=440.0)
    >>> saw.get_item(0, 44100)
    0.0
"""

#
### Import Modules. ###
#
import math
from typing import Any

#
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant


#
class BandLimitedSawtooth(Value):
    """A "good listening" sawtooth wave built from a fixed number of harmonics.

    This class avoids the "naive" sawtooth formula (which introduces high-frequency
    aliasing) by summing `num_harmonics` sine waves. This is an additive synthesis
    model for a band-limited oscillator.

    Attributes:
        time (Value): The time source.
        frequency (Value): The fundamental frequency in Hz.
        amplitude (Value): The peak amplitude.
        num_harmonics (int): Number of harmonics to sum.
    """

    #
    def __init__(
        self,
        time: Value,
        frequency: Value,  # Frequency in Hz
        amplitude: Value = Constant(1.0),
        num_harmonics: int = 15,
    ) -> None:
        """Initializes the BandLimitedSawtooth wave.

        Args:
            time (Value): The time source.
            frequency (Value): The fundamental frequency in Hz.
            amplitude (Value): The peak amplitude.
            num_harmonics (int): Number of harmonics to sum (default 15).
        """

        #
        super().__init__()

        #
        self.time: Value = time
        self.frequency: Value = (
            frequency if isinstance(frequency, Value) else Constant(frequency)
        )
        self.amplitude: Value = (
            amplitude if isinstance(amplitude, Value) else Constant(amplitude)
        )
        self.num_harmonics: int = max(1, num_harmonics)
        self.pi2: float = 2 * math.pi

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the wave value for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The wave amplitude at the given index.
        """

        #
        t_v: float = self.time.get_item(index=index, sample_rate=sample_rate)
        f_v: float = self.frequency.get_item(index=index, sample_rate=sample_rate)
        a_v: float = self.amplitude.get_item(index=index, sample_rate=sample_rate)

        #
        output: float = 0.0
        #
        for n in range(1, self.num_harmonics + 1):
            #
            phase: float = t_v * f_v * n * self.pi2
            #
            output += math.sin(phase) / n

        #
        ### Normalize (approx. 2/pi) and apply amplitude. ###
        #
        return output * 0.6366 * a_v

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the sawtooth wave.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized wave samples.
        """

        #
        t_v: NDArray[np.float32] = self.time.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        f_v: NDArray[np.float32] = self.frequency.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        a_v: NDArray[np.float32] = self.amplitude.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        output_array: NDArray[np.float32] = np.zeros_like(
            indexes_buffer, dtype=np.float32
        )

        #
        ### Sum the harmonics. ###
        #
        for n in range(1, self.num_harmonics + 1):
            #
            phase: NDArray[np.float32] = (t_v * f_v * n * self.pi2).astype(
                dtype=np.float32
            )
            #
            output_array += np.sin(phase) / n

        #
        ### Normalize (approx. 2/pi) and apply amplitude. ###
        #
        return (output_array * 0.6366 * a_v).astype(dtype=np.float32)

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
            Tensor: A tensor of wave samples.
        """

        #
        t_v: Tensor = self.time.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        f_v: Tensor = self.frequency.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        a_v: Tensor = self.amplitude.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        output_array: Tensor = torch.zeros_like(
            indexes_buffer, dtype=torch.float32, device=device
        )

        #
        ### Sum the harmonics. ###
        #
        for n in range(1, self.num_harmonics + 1):
            #
            phase: Tensor = (t_v * f_v * n * self.pi2).to(dtype=torch.float32)
            #
            output_array = output_array + (torch.sin(phase) / n)

        #
        ### Normalize (approx. 2/pi) and apply amplitude. ###
        #
        return (output_array * 0.6366 * a_v).to(dtype=torch.float32)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through additive synthesis.

        Gradients are computed for frequency, amplitude, and time by differentiating
        the sum of sines.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        t_v = self.time.getitem_np(np.zeros_like(grad_output), sample_rate)
        f_v = self.frequency.getitem_np(np.zeros_like(grad_output), sample_rate)
        a_v = self.amplitude.getitem_np(np.zeros_like(grad_output), sample_rate)

        common_sum_cos = np.zeros_like(grad_output)
        common_sum_sin_n = np.zeros_like(grad_output)

        for n in range(1, self.num_harmonics + 1):
            phase = t_v * f_v * n * self.pi2
            common_sum_cos += np.cos(phase)
            common_sum_sin_n += np.sin(phase) / n

        # dy/da = 0.6366 * sum(sin(phi)/n)
        self.amplitude.backward(
            grad_output * 0.6366 * common_sum_sin_n, context, sample_rate
        )

        # dphi/dt = 2pi * f * n
        # dphi/df = 2pi * t * n
        # dy/dt = 0.6366 * a * sum(cos(phi) * 2pi * f) = 0.6366 * a * 2pi * f * sum(cos(phi))
        grad_base = grad_output * 0.6366 * a_v * self.pi2 * common_sum_cos
        self.time.backward(grad_base * f_v, context, sample_rate)
        self.frequency.backward(grad_base * t_v, context, sample_rate)
