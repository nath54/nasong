from typing import Dict, Any
import math
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant


#
class Sawtooth(Value):
    """
    A Value that generates a "naive" sawtooth wave.
    """

    #
    def __init__(
        self,
        value: Value,
        frequency: Value = Constant(1),
        amplitude: Value = Constant(1),
        delta: Value = Constant(0),
        direction: Value = Constant(1),  # 1 for rising, -1 for falling
    ) -> None:
        """
        Initializes the Sawtooth oscillator.

        Args:
            value: The input phase Value (e.g., time).
            frequency: The frequency multiplier.
            amplitude: The amplitude (gain).
            delta: The phase offset.
            direction: A Value determinations the slope.
                        >= 0 gives a rising sawtooth.
                        < 0 gives a falling sawtooth.
        """

        #
        super().__init__()

        #
        self.value: Value = value
        self.frequency: Value = frequency
        self.amplitude: Value = amplitude
        self.delta: Value = delta
        self.direction: Value = direction

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        val_v: float = self.value.get_item(index=index, sample_rate=sample_rate)
        fre_v: float = self.frequency.get_item(index=index, sample_rate=sample_rate)
        amp_v: float = self.amplitude.get_item(index=index, sample_rate=sample_rate)
        del_v: float = self.delta.get_item(index=index, sample_rate=sample_rate)
        dir_v: float = self.direction.get_item(index=index, sample_rate=sample_rate)

        #
        ### Calculate the phase and normalize to [0, 1). ###
        #
        phase: float = val_v * fre_v + del_v
        normalized_phase: float = phase - math.floor(phase)

        #
        ### Sawtooth wave: linear rise from -1 to 1 (or fall). ###
        #
        if dir_v >= 0:
            #
            ### Rising sawtooth: goes from -1 to 1. ###
            #
            sawtooth_value: float = 2.0 * normalized_phase - 1.0
        #
        else:
            #
            ### Falling sawtooth: goes from 1 to -1. ###
            #
            sawtooth_value: float = 1.0 - 2.0 * normalized_phase

        #
        ### Apply amplitude scaling. ###
        #
        return amp_v * sawtooth_value

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
        dir_v: NDArray[np.float32] = self.direction.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        ### Calculate the phase and normalize to [0, 1). ###
        #
        phase: NDArray[np.float32] = np.multiply(val_v, fre_v) + del_v
        normalized_phase: NDArray[np.float32] = phase - np.floor(phase)

        #
        ### Sawtooth wave: linear rise from -1 to 1 (or fall). ###
        #
        rising_sawtooth: NDArray[np.float32] = (2.0 * normalized_phase - 1.0).astype(
            dtype=np.float32
        )
        falling_sawtooth: NDArray[np.float32] = (1.0 - 2.0 * normalized_phase).astype(
            dtype=np.float32
        )
        sawtooth_value: NDArray[np.float32] = np.where(
            dir_v >= 0, rising_sawtooth, falling_sawtooth
        )

        #
        ### Apply amplitude scaling. ###
        #
        return np.multiply(amp_v, sawtooth_value)

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
        dir_v: Tensor = self.direction.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        ### Calculate the phase and normalize to [0, 1). ###
        #
        phase: Tensor = val_v * fre_v + del_v
        normalized_phase: Tensor = phase - torch.floor(phase)

        #
        ### Sawtooth wave: linear rise from -1 to 1 (or fall). ###
        #
        rising_sawtooth: Tensor = (2.0 * normalized_phase - 1.0).to(dtype=torch.float32)
        falling_sawtooth: Tensor = (1.0 - 2.0 * normalized_phase).to(
            dtype=torch.float32
        )
        sawtooth_value: Tensor = torch.where(
            dir_v >= 0, rising_sawtooth, falling_sawtooth
        )

        #
        ### Apply amplitude scaling. ###
        #
        return amp_v * sawtooth_value

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients through sawtooth wave.
        Ignoring the discontinuity (jump).
        Rising: y = a * (2 * p - 1) where p = (v * f + d) % 1
        dy/da = 2*p - 1
        dy/dp = 2*a
        dp/dv = f, dp/df = v, dp/dd = 1
        Falling: y = a * (1 - 2 * p)
        dy/dp = -2*a
        """
        val_v = self.value.getitem_np(context["indices"], sample_rate)
        f_v = self.frequency.getitem_np(context["indices"], sample_rate)
        amp_v = self.amplitude.getitem_np(context["indices"], sample_rate)
        d_v = self.delta.getitem_np(context["indices"], sample_rate)
        dir_v = self.direction.getitem_np(np.zeros_like(grad_output), sample_rate)

        phase = val_v * f_v + d_v
        p = phase - np.floor(phase)

        # dy/da
        saw_val = np.where(dir_v >= 0, 2.0 * p - 1.0, 1.0 - 2.0 * p)
        self.amplitude.backward(grad_output * saw_val, context, sample_rate)

        # dy/dp
        slope = np.where(dir_v >= 0, 2.0, -2.0)
        grad_base = grad_output * amp_v * slope
        self.value.backward(grad_base * f_v, context, sample_rate)
        self.frequency.backward(grad_base * val_v, context, sample_rate)
        self.delta.backward(grad_base, context, sample_rate)
