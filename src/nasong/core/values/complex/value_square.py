from typing import Dict, Any
import math
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant


#
class Square(Value):
    """
    A Value that generates a "naive" square wave with a variable duty cycle.
    """

    #
    def __init__(
        self,
        value: Value,
        frequency: Value = Constant(1),
        amplitude: Value = Constant(1),
        delta: Value = Constant(0),
        duty_cycle: Value = Constant(0.5),
    ) -> None:
        """
        Initializes the Square oscillator.

        Args:
            value: The input phase Value (e.g., time).
            frequency: The frequency multiplier.
            amplitude: The amplitude (gain).
            delta: The phase offset.
            duty_cycle: The fraction of the period (0.0 to 1.0) for
                        which the signal is high.
        """

        #
        super().__init__()

        #
        self.value: Value = value
        self.frequency: Value = (
            frequency if isinstance(frequency, Value) else Constant(frequency)
        )
        self.amplitude: Value = (
            amplitude if isinstance(amplitude, Value) else Constant(amplitude)
        )
        self.delta: Value = delta if isinstance(delta, Value) else Constant(delta)
        self.duty_cycle: Value = (
            duty_cycle if isinstance(duty_cycle, Value) else Constant(duty_cycle)
        )

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        val_v: float = self.value.get_item(index=index, sample_rate=sample_rate)
        fre_v: float = self.frequency.get_item(index=index, sample_rate=sample_rate)
        amp_v: float = self.amplitude.get_item(index=index, sample_rate=sample_rate)
        del_v: float = self.delta.get_item(index=index, sample_rate=sample_rate)
        duty_v: float = self.duty_cycle.get_item(index=index, sample_rate=sample_rate)

        #
        ### Calculate the phase and normalize to [0, 1). ###
        #
        phase: float = val_v * fre_v + del_v
        normalized_phase: float = phase - math.floor(phase)

        #
        ### Square wave: high for duty_cycle portion, low for the rest. ###
        #
        if normalized_phase < duty_v:
            #
            return amp_v
        #
        else:
            #
            return -amp_v

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
        duty_v: NDArray[np.float32] = self.duty_cycle.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        ### Calculate the phase and normalize to [0, 1). ###
        #
        phase: NDArray[np.float32] = np.multiply(val_v, fre_v) + del_v
        normalized_phase: NDArray[np.float32] = phase - np.floor(phase)

        #
        ### Square wave: high for duty_cycle portion, low for the rest. ###
        #
        return np.where(normalized_phase < duty_v, amp_v, -amp_v)

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
        duty_v: Tensor = self.duty_cycle.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        ### Calculate the phase and normalize to [0, 1). ###
        #
        phase: Tensor = val_v * fre_v + del_v
        normalized_phase: Tensor = phase - torch.floor(phase)

        #
        ### Square wave: high for duty_cycle portion, low for the rest. ###
        #
        return torch.where(normalized_phase < duty_v, amp_v, -amp_v)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients through square wave.
        y = a * sign(duty - (v*f + d)%1) (roughly)
        dy/da = sign(...)
        dy/dduty = 2 * a * delta(duty - p)
        For simplicity, we use straight-through: dy/dduty = 0 almost everywhere,
        or we can use a narrow pulse. Here we'll just handle dy/da.
        """
        val_v = self.value.getitem_np(context["indices"], sample_rate)
        f_v = self.frequency.getitem_np(context["indices"], sample_rate)
        amp_v = self.amplitude.getitem_np(context["indices"], sample_rate)
        d_v = self.delta.getitem_np(context["indices"], sample_rate)
        duty = self.duty_cycle.getitem_np(context["indices"], sample_rate)

        phase = val_v * f_v + d_v
        p = phase - np.floor(phase)

        # dy/da
        sq_val = np.where(p < duty, 1.0, -1.0)
        self.amplitude.backward(grad_output * sq_val, context, sample_rate)

        # dy/dduty: straight-through proxy
        # We can approximate dy/dduty = a * pulse at the transition
        # But for now, we leave it at 0 as per the comment.
        self.duty_cycle.backward(np.zeros_like(grad_output), context, sample_rate)

        # Straight-through for phase-related grads
        # Use a sawtooth-like proxy: ramp from -1 to 1 per cycle
        grad_proxy = grad_output * amp_v * 2.0
        self.value.backward(grad_proxy * f_v, context, sample_rate)
        self.frequency.backward(grad_proxy * val_v, context, sample_rate)
        self.delta.backward(grad_proxy, context, sample_rate)
