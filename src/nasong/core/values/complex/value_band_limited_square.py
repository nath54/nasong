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
TODO: add full docstring, explaining what the goal of this script is, and explaining for each class and each function what is it, how it works, and how to use it.
"""

#
### Import Modules. ###
#
from typing import Dict, Any
import math
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant


#
class BandLimitedSquare(Value):
    """
    A "good listening" square wave built from a fixed number of harmonics.
    """

    #
    def __init__(
        self,
        time: Value,
        frequency: Value,  # Frequency in Hz
        amplitude: Value = Constant(1.0),
        num_harmonics: int = 15,
    ) -> None:

        #
        super().__init__()

        #
        self.time: Value = time
        self.frequency: Value = frequency
        self.amplitude: Value = amplitude
        self.num_harmonics: int = max(1, num_harmonics)
        self.pi2: float = 2 * math.pi

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        t_v: float = self.time.get_item(index=index, sample_rate=sample_rate)
        f_v: float = self.frequency.get_item(index=index, sample_rate=sample_rate)
        a_v: float = self.amplitude.get_item(index=index, sample_rate=sample_rate)

        #
        output: float = 0.0
        #
        for n in range(1, self.num_harmonics + 1):
            #
            harmonic: int = 2 * n - 1  # Only odd harmonics
            #
            phase: float = t_v * f_v * harmonic * self.pi2
            #
            output += math.sin(phase) / harmonic

        #
        ### Normalize (approx. 4/pi) and apply amplitude. ###
        #
        return output * 0.7854 * a_v

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

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
            harmonic: int = 2 * n - 1  # Only odd harmonics
            #
            phase: NDArray[np.float32] = (t_v * f_v * harmonic * self.pi2).astype(
                dtype=np.float32
            )
            #
            output_array += np.sin(phase) / harmonic

        #
        ### Normalize (approx. 4/pi) and apply amplitude. ###
        #
        return (output_array * 0.7854 * a_v).astype(dtype=np.float32)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

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
            harmonic: int = 2 * n - 1  # Only odd harmonics
            #
            phase: Tensor = (t_v * f_v * harmonic * self.pi2).to(dtype=torch.float32)
            #
            output_array = output_array + (torch.sin(phase) / harmonic)

        #
        ### Normalize (approx. 4/pi) and apply amplitude. ###
        #
        return (output_array * 0.7854 * a_v).to(dtype=torch.float32)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients through band-limited square wave.
        y = 0.7854 * a * sum(sin(t * f * h * 2pi) / h)
        dy/da = 0.7854 * sum(...)
        dy/dt = 0.7854 * a * sum(cos(...) * f * h * 2pi / h) = 0.7854 * a * f * 2pi * sum(cos(...))
        dy/df = 0.7854 * a * t * 2pi * sum(cos(...))
        """
        t_v = self.time.getitem_np(np.zeros_like(grad_output), sample_rate)
        f_v = self.frequency.getitem_np(np.zeros_like(grad_output), sample_rate)
        a_v = self.amplitude.getitem_np(np.zeros_like(grad_output), sample_rate)

        sum_sin = np.zeros_like(grad_output)
        sum_cos = np.zeros_like(grad_output)

        for n in range(1, self.num_harmonics + 1):
            h = 2 * n - 1
            phase = t_v * f_v * h * self.pi2
            sum_sin += np.sin(phase) / h
            sum_cos += np.cos(phase)

        # dy/da
        self.amplitude.backward(grad_output * 0.7854 * sum_sin, context, sample_rate)

        # dy/dt and dy/df
        grad_base = grad_output * 0.7854 * a_v * self.pi2 * sum_cos
        self.time.backward(grad_base * f_v, context, sample_rate)
        self.frequency.backward(grad_base * t_v, context, sample_rate)
