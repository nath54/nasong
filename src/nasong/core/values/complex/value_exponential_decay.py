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
Exponential decay envelope implementation.

This module provides the `ExponentialDecay` class, which implements a
simple one-shot exponential decay envelope starting from a trigger time.

Example:
    >>> from nasong.core.values.basic.value_identity import Identity
    >>> from nasong.core.values.complex.value_exponential_decay import ExponentialDecay
    >>> time = Identity()
    >>> decay = ExponentialDecay(time=time, start_time=0.0, decay_rate=10.0)
    >>> decay.get_item(0, 44100)
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


#
class ExponentialDecay(Value):
    """A simple, one-shot exponential decay envelope.

    This class generates an envelope that starts at 1.0 at `start_time` and
    decays exponentially towards 0.0 at a rate defined by `decay_rate`.

    Attributes:
        time (Value): The time source.
        start_time (float): The time when the decay starts.
        decay_rate (float): The rate of decay (higher = faster decay).
    """

    #
    def __init__(
        self,
        time: Value,
        start_time: float,
        decay_rate: float = 15.0,  # e.g., 15 for snare, 8 for kick
    ) -> None:
        """Initializes the ExponentialDecay envelope.

        Args:
            time (Value): The time source.
            start_time (float): The trigger time.
            decay_rate (float): The rate of exponential decay (default 15.0).
        """

        #
        super().__init__()

        #
        self.time: Value = time
        self.start_time: float = start_time
        self.decay_rate: float = decay_rate

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the decay value for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The decay amplitude at the given index.
        """

        #
        t: float = self.time.get_item(index=index, sample_rate=sample_rate)
        relative_time: float = t - self.start_time

        #
        ### Gate: before the note, output 0. ###
        #
        if relative_time < 0:
            #
            return 0.0

        #
        ### Exponential decay. ###
        #
        return math.exp(-relative_time * self.decay_rate)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the decay envelope.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized envelope samples.
        """

        #
        t: NDArray[np.float32] = self.time.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        #
        relative_time: NDArray[np.float32] = t - self.start_time

        #
        ### Gate: Create a mask for all samples at or after the start. ###
        #
        gate_mask: NDArray[np.bool_] = relative_time >= 0

        #
        if not np.any(gate_mask):
            #
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        safe_relative_time: NDArray[np.float32] = np.maximum(0.0, relative_time)

        #
        ### Calculate decay for all samples using the safe time. ###
        #
        decay_envelope: NDArray[np.float32] = np.exp(
            -safe_relative_time * self.decay_rate
        ).astype(dtype=np.float32)

        #
        ### Apply gate mask to zero out samples before the start_time. ###
        #
        return decay_envelope * gate_mask

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Generates the decay envelope for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of envelope samples.
        """

        #
        t: Tensor = self.time.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        #
        relative_time: Tensor = t - self.start_time

        #
        ### Gate: Create a mask for all samples at or after the start. ###
        #
        gate_mask: Tensor = (relative_time >= 0).to(dtype=torch.float32)

        #
        if not torch.any(gate_mask):
            #
            return torch.zeros_like(indexes_buffer, dtype=torch.float32, device=device)

        #
        ### Calculate decay for all samples using the safe time. ###
        #
        decay_envelope: Tensor = torch.exp(-relative_time * self.decay_rate).to(
            dtype=torch.float32
        )

        #
        ### Apply gate mask to zero out samples before the start_time. ###
        #
        return decay_envelope * gate_mask

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the exponential decay.

        Computes gradients for the base time source by differentiating the
        exponential decay function.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        t = self.time.getitem_np(np.zeros_like(grad_output), sample_rate)
        relative_time = t - self.start_time

        gate_mask = relative_time >= 0
        y = np.exp(-np.maximum(0.0, relative_time) * self.decay_rate)

        dy_dt = -self.decay_rate * y

        self.time.backward(grad_output * dy_dt * gate_mask, context, sample_rate)
