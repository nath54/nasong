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
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor


#
class WhiteNoise(Value):
    #
    def __init__(self, seed: int = 42, scale: float = 1.0) -> None:

        #
        super().__init__()

        #
        self.seed: int = seed
        #
        self.scale: float = scale

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        # We use a non-vectorized version for single samples if needed,
        # but internal code usually uses getitem_np.
        # Simple hash-based approach for single items:
        noise = float(((index * self.seed + 12345) & 0xFFFFFFFF) / 0xFFFFFFFF - 0.5)
        return noise * 100.0 * self.scale

    #
    ### Helper function for vectorized deterministic noise ###
    #
    @staticmethod
    def vectorized_noise(
        indexes_buffer: NDArray[np.float32], seed: int, scale: float
    ) -> NDArray[np.float32]:
        """
        Generates a deterministic, pseudo-random noise value for each index.
        """

        #
        idx_int: NDArray[np.uint32] = indexes_buffer.astype(np.uint32)
        noise_int: NDArray[np.uint32] = ((idx_int * seed + 12345) & 0xFFFFFFFF).astype(
            dtype=np.uint32
        )

        #
        ### Convert to float in range [-0.5, 0.5] ###
        #
        noise_float: NDArray[np.float32] = (
            (noise_int.astype(np.float32) / 0xFFFFFFFF) - 0.5
        ).astype(dtype=np.float32)

        #
        ### Scale to match original intent (e.g., approx -50 to 50, then / 5000) ###
        #
        return (noise_float * 100.0 * scale).astype(dtype=np.float32)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        #
        return self.__class__.vectorized_noise(indexes_buffer, self.seed, self.scale)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """
        Generates deterministic, differentiable noise for training.
        Uses LCG (Linear Congruential Generator) approach.
        """

        #
        ### Convert indexes to int32 for LCG. ###
        #
        idx_int: Tensor = indexes_buffer.to(torch.int32)
        noise_int: Tensor = ((idx_int * self.seed + 12345) & 0xFFFFFFFF).to(torch.int32)

        #
        ### Convert to float in range [-0.5, 0.5]. ###
        #
        noise_float: Tensor = (noise_int.to(torch.float32) / 0xFFFFFFFF) - 0.5

        #
        ### Scale to match original intent. ###
        #
        return (noise_float * 100.0 * self.scale).to(dtype=torch.float32, device=device)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        WhiteNoise has no upstream trainable parameters in current implementation.
        """
        pass
