#
### Import Modules. ###
#

#
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant


#
class LowPass(Value):
    """
    A simple "clipper" Value that limits the maximum value.
    This is NOT an audio filter (like a Butterworth or RC filter).
    It is equivalent to min(value, max_value).
    """

    #
    def __init__(self, value: Value, max_value: Value = Constant(0)) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.max_value: Value = max_value

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        return min(
            self.max_value.get_item(index=index, sample_rate=sample_rate),
            self.value.get_item(index=index, sample_rate=sample_rate),
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        return np.minimum(
            self.max_value.getitem_np(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate
            ),
            self.value.getitem_np(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate
            ),
        )

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        return torch.minimum(
            self.max_value.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            ),
            self.value.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            ),
        )
