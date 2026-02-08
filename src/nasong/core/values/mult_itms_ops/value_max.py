from typing import Dict, Any
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.input_args import input_args_to_values


#
class Max(Value):
    """A Value that returns the maximum value from a list of input Values."""

    #
    def __init__(self, *values: Value | list[Value]) -> None:

        #
        super().__init__()

        #
        self.values: list[Value] = input_args_to_values(values=values)

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        return max(
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

        # Autograd fix: np.maximum.reduce is not supported, use stack + max.
        stacked = np.stack(arrays, axis=0)
        return np.max(stacked, axis=0)

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
            return torch.max(stacked, dim=0)[0]

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients through max.
        y = max(x_i)
        dy/dx_i = 1 if x_i is max, 0 otherwise.
        """
        if not self.values:
            return

        arrays = [
            v.getitem_np(np.zeros_like(grad_output), sample_rate) for v in self.values
        ]
        stacked = np.stack(arrays, axis=0)
        max_idx = np.argmax(stacked, axis=0)

        for i, v in enumerate(self.values):
            mask = (max_idx == i).astype(np.float32)
            v.backward(grad_output * mask, context, sample_rate)
