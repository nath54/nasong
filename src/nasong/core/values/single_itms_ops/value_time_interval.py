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
class TimeInterval(Value):
    """
    A Value that selects between two other Values based on the index (time).
    Returns `value_inside` if `min_sample_idx <= index <= max_sample_idx`.
    Otherwise, returns `value_outside`.
    """

    #
    def __init__(
        self,
        value_inside: Value,
        value_outside: Value = Constant(0),
        min_sample_idx: Value = Constant(0),
        max_sample_idx: Value = Constant(1),
    ) -> None:

        #
        super().__init__()

        #
        self.value_inside: Value = value_inside
        self.value_outside: Value = value_outside
        self.min_sample_idx: Value = min_sample_idx
        self.max_sample_idx: Value = max_sample_idx

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        if index < self.min_sample_idx.get_item(index=index, sample_rate=sample_rate):
            #
            return self.value_outside.get_item(index=index, sample_rate=sample_rate)

        #
        elif index > self.max_sample_idx.get_item(index=index, sample_rate=sample_rate):
            #
            return self.value_outside.get_item(index=index, sample_rate=sample_rate)

        #
        return self.value_inside.get_item(index=index, sample_rate=sample_rate)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        inside_values: NDArray[np.float32] = self.value_inside.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        outside_values: NDArray[np.float32] = self.value_outside.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        min_idx: NDArray[np.float32] = self.min_sample_idx.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        max_idx: NDArray[np.float32] = self.max_sample_idx.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        ### Create mask for values inside the interval. ###
        #
        inside_mask = (indexes_buffer >= min_idx) & (indexes_buffer <= max_idx)

        #
        return np.where(inside_mask, inside_values, outside_values)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        inside_values: Tensor = self.value_inside.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        outside_values: Tensor = self.value_outside.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        min_idx: Tensor = self.min_sample_idx.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        max_idx: Tensor = self.max_sample_idx.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        ### Create mask for values inside the interval. ###
        #
        inside_mask = (indexes_buffer >= min_idx) & (indexes_buffer <= max_idx)

        #
        return torch.where(inside_mask, inside_values, outside_values)
