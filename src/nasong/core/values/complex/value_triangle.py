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
class Triangle(Value):
    """
    A Value that generates a "naive" triangle wave.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": This is a mathematically correct, "naive" triangle
            wave. It is "truthful" in that sense.
        - "Good Listening": This implementation is **POOR** for "good listening"
            when used for audio-rate oscillators (e.g., frequencies > 20 Hz).
        - **Reason:** It produces strong aliasing (unwanted, inharmonic
            frequencies) because it has infinite sharp corners (harmonics)
            that are not band-limited. This aliasing sounds like a harsh,
            "digital" noise.
        - **Good Use:** This implementation is perfectly "realistic" and "good"
            for LFO (Low-Frequency Oscillator) use, where aliasing is not
            in the audible range.
    """

    #
    def __init__(
        self,
        value: Value,
        frequency: Value = Constant(1),
        amplitude: Value = Constant(1),
        delta: Value = Constant(0),
    ) -> None:
        """
        Initializes the Triangle oscillator.

        Args:
            value: The input phase Value (e.g., time).
            frequency: The frequency multiplier.
            amplitude: The amplitude (gain).
            delta: The phase offset.
        """

        #
        super().__init__()

        #
        self.value: Value = value
        self.frequency: Value = frequency
        self.amplitude: Value = amplitude
        self.delta: Value = delta

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        val_v: float = self.value.get_item(index=index, sample_rate=sample_rate)
        fre_v: float = self.frequency.get_item(index=index, sample_rate=sample_rate)
        amp_v: float = self.amplitude.get_item(index=index, sample_rate=sample_rate)
        del_v: float = self.delta.get_item(index=index, sample_rate=sample_rate)

        #
        ### Calculate the phase. ###
        #
        phase: float = val_v * fre_v + del_v

        #
        ### Triangle wave formula: 2 * |2 * (phase - floor(phase + 0.5))| - 1 ###
        ### This creates a wave that oscillates between -1 and 1              ###
        #
        triangle_value: float = 2.0 * abs(2.0 * (phase - math.floor(phase + 0.5))) - 1.0

        #
        ### Apply amplitude scaling. ###
        #
        return amp_v * triangle_value

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
        ### Calculate the phase. ###
        #
        phase: NDArray[np.float32] = np.multiply(val_v, fre_v) + del_v

        #
        ### Triangle wave formula: 2 * |2 * (phase - floor(phase + 0.5))| - 1 ###
        #
        triangle_value: NDArray[np.float32] = (
            2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0
        ).astype(dtype=np.float32)

        #
        ### Apply amplitude scaling. ###
        #
        return np.multiply(amp_v, triangle_value)

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
        ### Calculate the phase. ###
        #
        phase: Tensor = val_v * fre_v + del_v

        #
        ### Triangle wave formula: 2 * |2 * (phase - floor(phase + 0.5))| - 1 ###
        #
        triangle_value: Tensor = (
            2.0 * torch.abs(2.0 * (phase - torch.floor(phase + 0.5))) - 1.0
        ).to(dtype=torch.float32)

        #
        ### Apply amplitude scaling. ###
        #
        return amp_v * triangle_value
