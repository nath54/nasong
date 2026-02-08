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
class Identity(Value):
    """A Value that returns the sample index itself as the value."""

    #
    def __init__(self) -> None:

        #
        super().__init__()

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        return float(index)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        return indexes_buffer

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        return indexes_buffer.to(device)
