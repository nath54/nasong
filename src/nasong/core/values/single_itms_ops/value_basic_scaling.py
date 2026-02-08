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
class BasicScaling(Value):
    """A Value that applies a linear transformation: value * mult_scale + sum_scale."""

    #
    def __init__(self, value: Value, mult_scale: Value, sum_scale: Value) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.mult_scale: Value = mult_scale
        self.sum_scale: Value = sum_scale

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        v: float = self.value.get_item(index=index, sample_rate=sample_rate)
        m: float = self.mult_scale.get_item(index=index, sample_rate=sample_rate)
        s: float = self.sum_scale.get_item(index=index, sample_rate=sample_rate)

        #
        return v * m + s

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        v: NDArray[np.float32] = self.value.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        m: NDArray[np.float32] = self.mult_scale.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        s: NDArray[np.float32] = self.sum_scale.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        return np.multiply(v, m) + s

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        v: Tensor = self.value.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        m: Tensor = self.mult_scale.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        s: Tensor = self.sum_scale.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        return v * m + s
