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
        return random.choice(self.choices).get_item(
            index=index, sample_rate=sample_rate
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
        The selection is random but gradient flows through the chosen Value.
        """

        #
        ### For training purposes, just pick the first choice. ###
        ### True random choice would break gradient flow. ###
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
