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
        return np.random.uniform(
            low=min_vals, high=max_vals, size=indexes_buffer.shape
        ).astype(np.float32)

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
