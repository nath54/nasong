# Copyright (C) 2026 Nathan Cerisara <https://github.com/nath54/nasong>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""
TODO: add full docstring, explaining what the goal of this script is, and explaining for each class and each function what is it, how it works, and how to use it.
"""

#
### Import Modules. ###
#
from typing import Any
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
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradient to the first choice (consistent with getitem_np/torch).
        """
        if self.choices:
            self.choices[0].backward(grad_output, context, sample_rate)
