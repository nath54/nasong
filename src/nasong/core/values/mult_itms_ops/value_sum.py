#
### Import Modules. ###
#

#
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.input_args import input_args_to_values


#
class Sum(Value):
    """A Value that returns the sum of a list of input Values."""

    #
    def __init__(self, *values: Value | list[Value]) -> None:

        #
        super().__init__()

        #
        self.values: list[Value] = input_args_to_values(values=values)

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        return sum(
            [v.get_item(index=index, sample_rate=sample_rate) for v in self.values]
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        arrays = [
            v.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
            for v in self.values
        ]
        #
        return np.sum(arrays, axis=0)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        ### Compute all values and stack them. ###
        #
        value_tensors: list[Tensor] = [
            val.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            )
            for val in self.values
        ]

        #
        if len(value_tensors) == 0:
            #
            return torch.zeros_like(indexes_buffer, dtype=torch.float32, device=device)
        #
        elif len(value_tensors) == 1:
            #
            return value_tensors[0]
        #
        else:
            #
            stacked: Tensor = torch.stack(value_tensors, dim=0)
            #
            return torch.sum(stacked, dim=0)
