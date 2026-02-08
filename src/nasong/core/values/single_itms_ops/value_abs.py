#
### Import Modules. ###
#

#
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor


#
class Abs(Value):
    """A Value that returns the absolute value of another Value."""

    #
    def __init__(self, value: Value) -> None:

        #
        super().__init__()

        #
        self.value: Value = value

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        return abs(self.value.get_item(index=index, sample_rate=sample_rate))

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        return np.abs(
            self.value.getitem_np(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate
            )
        )

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        return torch.abs(
            self.value.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            )
        )
