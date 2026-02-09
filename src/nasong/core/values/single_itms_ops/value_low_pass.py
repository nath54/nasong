from typing import Dict, Any
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant


#
class LowPass(Value):
    """
    A simple "clipper" Value that limits the maximum value.
    This is NOT an audio filter (like a Butterworth or RC filter).
    It is equivalent to min(value, max_value).
    """

    #
    def __init__(self, value: Value, max_value: Value = Constant(0)) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.max_value: Value = (
            max_value if isinstance(max_value, Value) else Constant(max_value)
        )

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        return min(
            self.max_value.get_item(index=index, sample_rate=sample_rate),
            self.value.get_item(index=index, sample_rate=sample_rate),
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        return np.minimum(
            self.max_value.getitem_np(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate
            ),
            self.value.getitem_np(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate
            ),
        )

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        return torch.minimum(
            self.max_value.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            ),
            self.value.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            ),
        )

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients through low-pass (min clipper).
        y = min(x, high)
        dy/dx = 1 if x < high, 0 otherwise.
        dy/dhigh = 1 if high <= x, 0 otherwise.
        """
        v_val = self.value.getitem_np(context["indices"], sample_rate)
        high_val = self.max_value.getitem_np(context["indices"], sample_rate)

        mask_x = (v_val < high_val).astype(np.float32)
        mask_high = (high_val <= v_val).astype(np.float32)

        self.value.backward(grad_output * mask_x, context, sample_rate)
        self.max_value.backward(grad_output * mask_high, context, sample_rate)
