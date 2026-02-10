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
import random

#
import numpy as np
from numpy.typing import NDArray

from typing import Dict, Any

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor


#
class RandomFloat(Value):
    """
    A Value that returns a random float within a specified range
    for each sample.
    """

    #
    def __init__(self, min_range: Value, max_range: Value) -> None:

        #
        super().__init__()

        #
        self.min_range: Value = min_range
        self.max_range: Value = max_range

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        return random.uniform(
            a=float(self.min_range.get_item(index=index, sample_rate=sample_rate)),
            b=float(self.max_range.get_item(index=index, sample_rate=sample_rate)),
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """
        Returns a vectorized array of random floats.
        This is a performance-critical override.
        """

        #
        ### Get the vectorized min and max boundaries. ###
        #
        min_vals: NDArray[np.float32] = self.min_range.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        max_vals: NDArray[np.float32] = self.max_range.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        ### Use numpy's vectorized uniform random number generator. ###
        #
        random_vals: NDArray[np.float32] = np.random.uniform(
            low=0.0, high=1.0, size=indexes_buffer.shape
        ).astype(np.float32)

        # Save for backward pass
        self._last_random_vals = random_vals

        return min_vals + random_vals * (max_vals - min_vals)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """
        Returns differentiable random floats for training.
        Min/max are trainable.
        """

        #
        ### Get the vectorized min and max boundaries. ###
        #
        min_vals: Tensor = self.min_range.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        max_vals: Tensor = self.max_range.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        ### Use uniform random distribution with trainable bounds. ###
        #
        random_vals: Tensor = torch.rand_like(
            indexes_buffer, dtype=torch.float32, device=device
        )
        #
        return min_vals + random_vals * (max_vals - min_vals)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients to min_range and max_range.
        y = min + r*(max - min) = min*(1-r) + max*r
        dy/dmin = 1 - r
        dy/dmax = r
        """
        if hasattr(self, "_last_random_vals"):
            r = self._last_random_vals
            self.min_range.backward(grad_output * (1.0 - r), context, sample_rate)
            self.max_range.backward(grad_output * r, context, sample_rate)
