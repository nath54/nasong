#
### Import Modules. ###
#
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor


#
class Constant(Value):
    """A Value that returns the same constant number for all indexes."""

    #
    def __init__(self, value: float | int) -> None:

        #
        super().__init__()

        #
        self.value: float | int = value

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        return self.value

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        return np.full_like(indexes_buffer, fill_value=self.value, dtype=np.float32)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        return torch.full_like(
            indexes_buffer,
            fill_value=float(self.value),
            dtype=torch.float32,
            device=device,
        )


#
def c(v: float | int) -> Value:
    """Shorthand helper function to create a Constant Value."""

    #
    return Constant(value=v)
