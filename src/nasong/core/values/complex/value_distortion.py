#
### Import Modules. ###
#
import math

#
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor


#
class Distortion(Value):
    """
    Guitar distortion effect using `tanh` soft clipping.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": **EXCELLENT**. This is a "truthful" and classic
            model of a waveshaper used for soft-clipping distortion.
        - "Good Listening": **GOOD**. It does exactly what it's supposed to do:
            adds harmonics by clipping the waveform. The `tanh` function
            provides a "warm" sound compared to "hard" clipping.
    """

    #
    def __init__(self, value: Value, drive: float = 5.0) -> None:
        """
        Initializes the distortion effect.

        Args:
            value: The input `Value` (the audio signal) to be distorted.
            drive: The amount of gain to apply before clipping.
                    Higher values = more distortion.
        """

        #
        super().__init__()

        #
        self.value: Value = value
        self.drive: float = drive

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        ### Apply gain (drive). ###
        #
        x: float = (
            self.value.get_item(index=index, sample_rate=sample_rate) * self.drive
        )

        #
        ### Soft clipping using tanh. ###
        #
        return math.tanh(x) * 0.5

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        ### Get the input signal buffer and apply gain (drive). ###
        #
        x: NDArray[np.float32] = (
            self.value.getitem_np(
                indexes_buffer=indexes_buffer, sample_rate=sample_rate
            )
            * self.drive
        )

        #
        ### Apply vectorized soft clipping using np.tanh. ###
        #
        return (np.tanh(x) * 0.5).astype(dtype=np.float32)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        x: Tensor = self.value.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        gain: Tensor = self.gain.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        ### Amplify the signal. ###
        #
        x = x * gain

        #
        ### Apply vectorized soft clipping using torch.tanh. ###
        #
        return (torch.tanh(x) * 0.5).to(dtype=torch.float32)
