from typing import Dict, Any
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant


#
class Clamp(Value):
    """
    A Value that constrains another Value between a min and max Value.
    Also known as Clip.
    """

    #
    def __init__(
        self,
        value: Value,
        min_value: Value = Constant(0),
        max_value: Value = Constant(1),
    ) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.min_value: Value = min_value
        self.max_value: Value = max_value

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        return max(
            self.min_value.get_item(index=index, sample_rate=sample_rate),
            min(
                self.max_value.get_item(index=index, sample_rate=sample_rate),
                self.value.get_item(index=index, sample_rate=sample_rate),
            ),
        )

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        return np.clip(
            self.value.getitem_np(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate
            ),
            self.min_value.getitem_np(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate
            ),
            self.max_value.getitem_np(
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
        return torch.clamp(
            self.value.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            ),
            self.min_value.getitem_torch(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
            ),
            self.max_value.getitem_torch(
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
        Propagate gradients through clamp.
        y = clip(x, low, high)
        dy/dx = 1 if low < x < high, 0 otherwise.
        dy/dlow = 1 if x < low, 0 otherwise.
        dy/dhigh = 1 if x > high, 0 otherwise.
        """
        x_val = self.value.getitem_np(context["indices"], sample_rate)
        low_val = self.min_value.getitem_np(context["indices"], sample_rate)
        high_val = self.max_value.getitem_np(context["indices"], sample_rate)

        mask_x = (x_val > low_val) & (x_val < high_val)
        mask_low = x_val <= low_val
        mask_high = x_val >= high_val

        self.value.backward(
            grad_output * mask_x.astype(np.float32), context, sample_rate
        )
        self.min_value.backward(
            grad_output * mask_low.astype(np.float32), context, sample_rate
        )
        self.max_value.backward(
            grad_output * mask_high.astype(np.float32), context, sample_rate
        )
