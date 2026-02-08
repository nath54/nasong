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
class Modulo(Value):
    """
    A Value that computes the modulo (remainder) of another Value.
    Result = value % modulo_value

    This is ideal for creating looping LFOs (Low-Frequency Oscillators)
    by using a looping time value as the input to other oscillators
    like `Sawtooth` or `Triangle`.
    """

    #
    def __init__(self, value: Value, modulo_value: Value = Constant(1.0)) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.modulo_value: Value = modulo_value

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        val_v: float = self.value.get_item(index=index, sample_rate=sample_rate)
        mod_v: float = self.modulo_value.get_item(index=index, sample_rate=sample_rate)

        #
        ### Handle division by zero. ###
        #
        if mod_v == 0:
            #
            return val_v

        #
        ### Use % operator for correct "wrapping" behavior. ###
        #
        return val_v % mod_v

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        val_v: NDArray[np.float32] = self.value.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        mod_v: NDArray[np.float32] = self.modulo_value.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        ### Use np.mod for vectorized modulo. ###
        ### We use np.where to prevent division by zero. ###
        #
        return np.where(mod_v == 0, val_v, np.mod(val_v, mod_v))

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        val_v: Tensor = self.value.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        mod_v: Tensor = self.modulo_value.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        ### Use torch.fmod for vectorized modulo. ###
        ### We use torch.where to prevent division by zero. ###
        #
        return torch.where(mod_v == 0, val_v, torch.fmod(val_v, mod_v))
