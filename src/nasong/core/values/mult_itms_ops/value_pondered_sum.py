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
from nasong.core.values.input_args import input_args_to_values
from nasong.core.values.basic.value_constant import Constant


#
class PonderedSum(Value):
    """
    A Value that returns a weighted sum of (weight, value) pairs.
    Result = (weight1 * value1) + (weight2 * value2) + ...
    """

    #
    def __init__(self, values: list[tuple[Value, Value]]) -> None:

        #
        super().__init__()

        #
        self.values_and_weights: list[tuple[Value, Value]] = values

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        result: float = 0

        #
        for pond, val in self.values_and_weights:
            #
            result += pond.get_item(
                index=index, sample_rate=sample_rate
            ) * val.get_item(index=index, sample_rate=sample_rate)

        #
        return result

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        result: NDArray[np.float32] = np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        for pond, val in self.values_and_weights:
            #
            result += pond.getitem_np(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate
            ) * val.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        return result

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        if len(self.values_and_weights) == 0:
            #
            return torch.zeros_like(indexes_buffer, dtype=torch.float32, device=device)

        #
        ### Compute weighted sum. ###
        #
        result: Tensor = torch.zeros_like(
            indexes_buffer, dtype=torch.float32, device=device
        )

        #
        for weight, value in self.values_and_weights:
            #
            w_val: Tensor = weight.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            )
            v_val: Tensor = value.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            )
            #
            result = result + w_val * v_val

        #
        return result
