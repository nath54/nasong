from typing import Dict, Any
import math
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant


#
class Sin(Value):
    """
    A Value that generates a sine wave:
    amplitude * sin( (value * frequency) + delta )
    """

    #
    def __init__(
        self,
        value: Value,
        frequency: Value = Constant(1),
        amplitude: Value = Constant(1),
        delta: Value = Constant(0),
    ) -> None:

        #
        super().__init__()

        #
        # Ensure all inputs are Value objects
        self.value: Value = value
        self.frequency: Value = (
            frequency if isinstance(frequency, Value) else Constant(frequency)
        )
        self.amplitude: Value = (
            amplitude if isinstance(amplitude, Value) else Constant(amplitude)
        )
        self.delta: Value = delta if isinstance(delta, Value) else Constant(delta)

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        val_v: float = self.value.get_item(index=index, sample_rate=sample_rate)
        fre_v: float = self.frequency.get_item(index=index, sample_rate=sample_rate)
        amp_v: float = self.amplitude.get_item(index=index, sample_rate=sample_rate)
        del_v: float = self.delta.get_item(index=index, sample_rate=sample_rate)

        #
        return amp_v * math.sin(val_v * fre_v + del_v)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        val_v: NDArray[np.float32] = self.value.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        fre_v: NDArray[np.float32] = self.frequency.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        amp_v: NDArray[np.float32] = self.amplitude.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        del_v: NDArray[np.float32] = self.delta.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        return np.multiply(amp_v, np.sin(np.multiply(val_v, fre_v) + del_v))

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
        fre_v: Tensor = self.frequency.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        amp_v: Tensor = self.amplitude.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        del_v: Tensor = self.delta.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        return amp_v * torch.sin(val_v * fre_v + del_v)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients through Sine wave.
        y = a * sin(v * f + d)
        dy/da = sin(v * f + d)
        dy/dv = a * f * cos(v * f + d)
        dy/df = a * v * cos(v * f + d)
        dy/dd = a * cos(v * f + d)
        """
        val_v = self.value.getitem_np(context["indices"], sample_rate)
        fre_v = self.frequency.getitem_np(context["indices"], sample_rate)
        amp_v = self.amplitude.getitem_np(context["indices"], sample_rate)
        del_v = self.delta.getitem_np(context["indices"], sample_rate)

        phase = val_v * fre_v + del_v
        sin_phase = np.sin(phase)
        cos_phase = np.cos(phase)

        self.amplitude.backward(grad_output * sin_phase, context, sample_rate)

        grad_base = grad_output * amp_v * cos_phase
        self.value.backward(grad_base * fre_v, context, sample_rate)
        self.frequency.backward(grad_base * val_v, context, sample_rate)
        self.delta.backward(grad_base, context, sample_rate)
