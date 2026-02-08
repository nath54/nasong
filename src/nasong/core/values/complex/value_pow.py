#
### Import Modules. ###
#
import math

#
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant


#
class Pow(Value):
    """A Value that calculates base ^ exponent."""

    #
    def __init__(self, exponent: Value, base: Value = Constant(value=math.e)) -> None:

        #
        super().__init__()

        #
        self.exponent: Value = exponent
        self.base: Value = base

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        base_v: float = self.base.get_item(index=index, sample_rate=sample_rate)
        exp_v: float = self.exponent.get_item(index=index, sample_rate=sample_rate)

        #
        return base_v**exp_v

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        base_v: NDArray[np.float32] = self.base.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        exp_v: NDArray[np.float32] = self.exponent.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        return np.power(base_v, exp_v)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        base_v: Tensor = self.base.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        exp_v: Tensor = self.exponent.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        return torch.pow(base_v, exp_v)
