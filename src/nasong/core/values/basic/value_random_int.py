#
### Import Modules. ###
#

#
import random

#
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor


#
class RandomInt(Value):
    """
    A Value that returns a random integer within a specified range
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
        return float(
            random.randint(
                a=int(self.min_range.get_item(index=index, sample_rate=sample_rate)),
                b=int(self.max_range.get_item(index=index, sample_rate=sample_rate)),
            )
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """
        Returns a vectorized array of random integers (as floats).
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
        ### Vectorized version using numpy's random generator. ###
        #
        min_int: NDArray[np.int64] = min_vals.astype(np.int64)

        #
        ### np.random.randint's 'high' parameter is exclusive, so we add 1 ###
        ### to match the inclusive behavior of random.randint.             ###
        #
        max_int: NDArray[np.int64] = max_vals.astype(np.int64) + 1

        #
        ### Ensure 'high' is always strictly greater than 'low'. ###
        ### If min_int >= max_int, set max_int to min_int + 1.   ###
        #
        max_int = np.maximum(min_int + 1, max_int)

        #
        ### Generate random ints and cast back to float32 for the audio buffer. ###
        #
        return np.random.randint(
            low=min_int, high=max_int, size=indexes_buffer.shape
        ).astype(np.float32)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """
        Returns differentiable random values for training.
        Min/max are trainable, but the random selection itself is not.
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
        ### Use continuous uniform distribution (trainable bounds). ###
        ### Round to integers but use smooth random for gradient flow. ###
        #
        random_vals: Tensor = torch.rand_like(
            indexes_buffer, dtype=torch.float32, device=device
        )
        #
        ### Scale to range and round (rounding breaks gradient but bounds are trainable). ###
        #
        return torch.floor(min_vals + random_vals * (max_vals - min_vals + 1.0)).to(
            dtype=torch.float32, device=device
        )
