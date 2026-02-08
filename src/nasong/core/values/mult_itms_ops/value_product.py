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
class Product(Value):
    """A Value that returns the product of a list of input Values."""

    #
    def __init__(self, *values: Value | list[Value]) -> None:

        #
        super().__init__()

        #
        self.values: list[Value] = input_args_to_values(values=values)

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        result: float = 1

        #
        for v in self.values:
            #
            result *= v.get_item(index=index, sample_rate=sample_rate)

        #
        return result

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        result: NDArray[np.float32] = np.ones_like(indexes_buffer, dtype=np.float32)

        #
        for v in self.values:
            #
            result = np.multiply(
                result,
                v.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate),
            )

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
        result: Tensor = torch.ones_like(
            indexes_buffer, dtype=torch.float32, device=device
        )

        #
        for v in self.values:
            #
            result = result * v.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            )

        #
        return result
