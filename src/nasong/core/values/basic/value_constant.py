#
### Import Modules. ###
#
import numpy as np
from numpy.typing import NDArray

from typing import Dict, Any

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor


#
class Constant(Value):
    """A Value that returns the same constant number for all indexes."""

    #
    def __init__(self, value: float | int) -> None:

        #
        super().__init__()

        #
        self.value: float | int = value

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        return float(self.value)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        return np.full_like(
            indexes_buffer, fill_value=float(self.value), dtype=np.float32
        )

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        return torch.full_like(
            indexes_buffer,
            fill_value=float(self.value),
            dtype=torch.float32,
            device=device,
        )

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Constant has no inputs, so backward does nothing."""
        pass


#
def c(v: float | int) -> Value:
    """Shorthand helper function to create a Constant Value."""

    #
    return Constant(value=v)
