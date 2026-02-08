#
### Import Modules. ###
#
from typing import cast, Callable, Any

#
import random
import math

#
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant


#
class WhiteNoise(Value):
    #
    def __init__(self, seed: int = 42, scale: float = 1.0):

        #
        super().__init__()

        #
        self.seed: int = seed
        #
        self.scale: float = scale

    #
    ### Helper function for vectorized deterministic noise ###
    #
    @staticmethod
    def vectorized_noise(
        indexes_buffer: NDArray[np.float32], seed: int, scale: float
    ) -> NDArray[np.float32]:
        """
        Generates a deterministic, pseudo-random noise value for each index.

        This replaces the non-vectorizable, non-performant `hash()`-based
        noise in the original classes. This uses a simple, fast LCG (Linear
        Congruential Generator) which is "hash-like" and deterministic.

        Args:
            indexes_buffer: The buffer of sample indices.
            seed: An integer to vary the noise (e.g., 8191, 7919).
            scale: The final scaling factor (e.g., 1/5000.0).

        Returns:
            A NumPy array of noise values, one for each index.
        """

        #
        ### A simple LCG: (a * x + c) % m              ###
        ### We use bitwise-AND for a fast modulo 2^32. ###
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
