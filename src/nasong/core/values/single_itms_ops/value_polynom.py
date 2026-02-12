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
Polynomial operation implementation.

This module provides the `Polynom` class, which calculates a polynomial function
of an input `Value` object using a list of coefficient `Value` objects.

Example:
    >>> from nasong.core.values.basic.value_constant import Constant
    >>> from nasong.core.values.single_itms_ops.value_polynom import Polynom
    >>> x = Constant(2.0)
    >>> terms = [Constant(1.0), Constant(2.0), Constant(3.0)] # 1 + 2x + 3x^2
    >>> p = Polynom(x, terms)
    >>> p.get_item(0, 44100)
    17.0
"""

#
### Import Modules. ###
#
from typing import Any
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant


#
class Polynom(Value):
    """A Value that calculates a polynomial function: y = terms[0] + terms[1]*X + terms[2]*X^2 + ...

    Attributes:
        X (Value): The variable Value.
        terms (list[Value]): The list of coefficient Values, starting from degree 0.
    """

    #
    def __init__(
        self, X: Value, terms: list[Value] = [Constant(0), Constant(1)]
    ) -> None:
        """Initializes the Polynom operation.

        Args:
            X (Value): The input variable Value.
            terms (list[Value], optional): The list of coefficient Value objects.
                Defaults to [Constant(0), Constant(1)] (identity function).
        """

        #
        super().__init__()

        #
        self.X: Value = X
        #
        self.terms: list[Value] = terms

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the polynomial result for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The result of the polynomial evaluation.
        """

        #
        X_val: float = self.X.get_item(index=index, sample_rate=sample_rate)

        #
        return sum(
            [
                X_val**i * self.terms[i].get_item(index=index, sample_rate=sample_rate)
                for i in range(len(self.terms))
            ]
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the polynomial results.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized polynomial samples.
        """

        #
        X_val: NDArray[np.float32] = self.X.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        return np.sum(
            [
                np.multiply(
                    np.power(X_val, i),
                    self.terms[i].getitem_np(
                        indexes_buffer=indexes_buffer, sample_rate=sample_rate
                    ),
                )
                for i in range(len(self.terms))
            ],
            axis=0,
        )

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Generates the polynomial results for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of polynomial samples.
        """

        #
        X_val: Tensor = self.X.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        result: Tensor = torch.zeros_like(
            indexes_buffer, dtype=torch.float32, device=device
        )
        #
        for i in range(len(self.terms)):
            #
            term_val: Tensor = self.terms[i].getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            )
            #
            result = result + torch.pow(X_val, float(i)) * term_val

        #
        return result

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the polynomial operation.

        Computes gradients for the variable X and each coefficient term.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        X_val = self.X.getitem_np(context["indices"], sample_rate)

        grad_dX = np.zeros_like(grad_output)

        for i, term in enumerate(self.terms):
            a_i = term.getitem_np(context["indices"], sample_rate)
            # dy/da_i
            term.backward(grad_output * np.power(X_val, i), context, sample_rate)

            # dy/dX contribution
            if i > 0:
                grad_dX += i * a_i * np.power(X_val, i - 1)

        self.X.backward(grad_output * grad_dX, context, sample_rate)
