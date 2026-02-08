from typing import Dict, Any
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant


#
class Polynom(Value):
    """
    A Value that calculates a polynomial function:
    y = terms[0] + terms[1]*X + terms[2]*X^2 + ...
    """

    #
    def __init__(
        self, X: Value, terms: list[Value] = [Constant(0), Constant(1)]
    ) -> None:

        #
        super().__init__()

        #
        self.X: Value = X
        #
        self.terms: list[Value] = terms

    #
    def get_item(self, index: int, sample_rate: int) -> float:

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
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients through polynomial.
        y = sum(a_i * X^i)
        dy/dX = sum(i * a_i * X^(i-1))
        dy/da_i = X^i
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
