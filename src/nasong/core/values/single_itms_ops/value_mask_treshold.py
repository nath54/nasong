from typing import Dict, Any
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant


#
class MaskTreshold(Value):
    """
    A Value that acts as a switch:
    if mask >= treshold, return mask_value.
    Otherwise, return the original value.
    """

    #
    def __init__(
        self,
        value: Value,
        mask: Value,
        treshold_to_mask: Value = Constant(1),
        mask_value: Value = Constant(0),
    ) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.mask: Value = mask
        self.treshold_to_mask: Value = treshold_to_mask
        self.mask_value: Value = mask_value

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        mask_v: float = self.mask.get_item(index=index, sample_rate=sample_rate)

        #
        if mask_v >= self.treshold_to_mask.get_item(
            index=index, sample_rate=sample_rate
        ):
            #
            return self.mask_value.get_item(index=index, sample_rate=sample_rate)

        #
        else:
            #
            return self.value.get_item(index=index, sample_rate=sample_rate)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        base_value: NDArray[np.float32] = self.value.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        masked_value: NDArray[np.float32] = self.mask_value.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        mask_v: NDArray[np.float32] = self.mask.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        treshold_v: NDArray[np.float32] = self.treshold_to_mask.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        return np.where(mask_v < treshold_v, base_value, masked_value)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        base_value: Tensor = self.value.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        masked_value: Tensor = self.mask_value.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        mask_v: Tensor = self.mask.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        treshold_v: Tensor = self.treshold_to_mask.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        return torch.where(mask_v < treshold_v, base_value, masked_value)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients through mask threshold.
        y = value if mask < threshold else mask_value
        dy/dvalue = 1 if mask < threshold else 0
        dy/dmask_value = 1 if mask >= threshold else 0
        """
        mask_v = self.mask.getitem_np(context["indices"], sample_rate)
        tresh_v = self.treshold_to_mask.getitem_np(context["indices"], sample_rate)

        mask_below = (mask_v < tresh_v).astype(np.float32)
        mask_above = (mask_v >= tresh_v).astype(np.float32)

        self.value.backward(grad_output * mask_below, context, sample_rate)
        self.mask_value.backward(grad_output * mask_above, context, sample_rate)
        # Straight-through for mask and threshold
        self.mask.backward(np.zeros_like(grad_output), context, sample_rate)
        self.treshold_to_mask.backward(np.zeros_like(grad_output), context, sample_rate)
