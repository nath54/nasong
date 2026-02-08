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
class MaskTreshold(Value):
    """
    A Value that acts as a switch:
    if mask >= treshold, return mask_value.
    Otherwise, return the original value.
    """

    #
    def __init__(
        self,
        value: Value,
        mask: Value,
        treshold_to_mask: Value = Constant(1),
        mask_value: Value = Constant(0),
    ) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.mask: Value = mask
        self.treshold_to_mask: Value = treshold_to_mask
        self.mask_value: Value = mask_value

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        mask_v: float = self.mask.get_item(index=index, sample_rate=sample_rate)

        #
        if mask_v >= self.treshold_to_mask.get_item(
            index=index, sample_rate=sample_rate
        ):
            #
            return self.mask_value.get_item(index=index, sample_rate=sample_rate)

        #
        else:
            #
            return self.value.get_item(index=index, sample_rate=sample_rate)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        base_value: NDArray[np.float32] = self.value.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        masked_value: NDArray[np.float32] = self.mask_value.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        mask_v: NDArray[np.float32] = self.mask.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        treshold_v: NDArray[np.float32] = self.treshold_to_mask.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        return np.where(mask_v < treshold_v, base_value, masked_value)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        base_value: Tensor = self.value.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        masked_value: Tensor = self.mask_value.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        mask_v: Tensor = self.mask.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        treshold_v: Tensor = self.treshold_to_mask.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        return torch.where(mask_v < treshold_v, base_value, masked_value)
