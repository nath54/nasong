from typing import Dict, Any
import random
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor


#
class RandomChoice(Value):
    """A Value that randomly selects from a list of other Value objects."""

    #
    def __init__(self, choices: list[Value]) -> None:

        #
        super().__init__()

        #
        self.choices: list[Value] = choices

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        if not self.choices:
            return 0.0

        return random.choice(self.choices).get_item(
            index=index, sample_rate=sample_rate
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """
        For NumPy rendering, we choose the first choice to maintain consistency with Torch.
        """
        if not self.choices:
            return np.zeros_like(indexes_buffer)

        return self.choices[0].getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """
        Returns a random choice for training.
        The selection is fixed to the first choice to maintain stable gradient flow.
        """

        #
        if len(self.choices) > 0:
            #
            return self.choices[0].getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            )
        #
        else:
            #
            return torch.zeros_like(indexes_buffer, dtype=torch.float32, device=device)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradient to the first choice (consistent with getitem_np/torch).
        """
        if self.choices:
            self.choices[0].backward(grad_output, context, sample_rate)
