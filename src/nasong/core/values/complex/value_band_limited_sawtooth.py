#
### Import Modules. ###
#
import math
from typing import Dict, Any

#
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant


#
class BandLimitedSawtooth(Value):
    """
        A "good listening" sawtooth wave built from a fixed number of harmonics.

        "Truthness" / "Good Listening" Analysis:
            - "Truthness": This is a "truthful" model of additive synthesis
                used to create a band-limited sawtooth wave.
            - "Good Listening": **GOOD**.
            - **Reason:** This class avoids the "naive" formula by summing
                `Sin` waves, following the model of the `WobbleBass` class
    .
                This is a "fixed-harmonic-limit" oscillator.
            - **Compromise:** This is not *perfectly* band-limited (which
                would require checking `frequency * n` against `sample_rate`
                for every sample). Instead, it uses a fixed `num_harmonics`,
                which is a "good listening" compromise that is vectorizable
                and supports dynamic frequency (e.g., vibrato).
    """

    #
    def __init__(
        self,
        time: Value,
        frequency: Value,  # Frequency in Hz
        amplitude: Value = Constant(1.0),
        num_harmonics: int = 15,
    ) -> None:

        #
        super().__init__()

        #
        self.time: Value = time
        self.frequency: Value = frequency
        self.amplitude: Value = amplitude
        self.num_harmonics: int = max(1, num_harmonics)
        self.pi2: float = 2 * math.pi

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        t_v: float = self.time.get_item(index=index, sample_rate=sample_rate)
        f_v: float = self.frequency.get_item(index=index, sample_rate=sample_rate)
        a_v: float = self.amplitude.get_item(index=index, sample_rate=sample_rate)

        #
        output: float = 0.0
        #
        for n in range(1, self.num_harmonics + 1):
            #
            phase: float = t_v * f_v * n * self.pi2
            #
            output += math.sin(phase) / n

        #
        ### Normalize (approx. 2/pi) and apply amplitude. ###
        #
        return output * 0.6366 * a_v

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        t_v: NDArray[np.float32] = self.time.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        f_v: NDArray[np.float32] = self.frequency.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        a_v: NDArray[np.float32] = self.amplitude.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        output_array: NDArray[np.float32] = np.zeros_like(
            indexes_buffer, dtype=np.float32
        )

        #
        ### Sum the harmonics. ###
        #
        for n in range(1, self.num_harmonics + 1):
            #
            phase: NDArray[np.float32] = (t_v * f_v * n * self.pi2).astype(
                dtype=np.float32
            )
            #
            output_array += np.sin(phase) / n

        #
        ### Normalize (approx. 2/pi) and apply amplitude. ###
        #
        return (output_array * 0.6366 * a_v).astype(dtype=np.float32)

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        t_v: Tensor = self.time.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        f_v: Tensor = self.frequency.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        a_v: Tensor = self.amplitude.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

        #
        output_array: Tensor = torch.zeros_like(
            indexes_buffer, dtype=torch.float32, device=device
        )

        #
        ### Sum the harmonics. ###
        #
        for n in range(1, self.num_harmonics + 1):
            #
            phase: Tensor = (t_v * f_v * n * self.pi2).to(dtype=torch.float32)
            #
            output_array = output_array + (torch.sin(phase) / n)

        #
        ### Normalize (approx. 2/pi) and apply amplitude. ###
        #
        return (output_array * 0.6366 * a_v).to(dtype=torch.float32)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients through additive synthesis.
        """
        t_v = self.time.getitem_np(np.zeros_like(grad_output), sample_rate)
        f_v = self.frequency.getitem_np(np.zeros_like(grad_output), sample_rate)
        a_v = self.amplitude.getitem_np(np.zeros_like(grad_output), sample_rate)

        common_sum_cos = np.zeros_like(grad_output)
        common_sum_sin_n = np.zeros_like(grad_output)

        for n in range(1, self.num_harmonics + 1):
            phase = t_v * f_v * n * self.pi2
            common_sum_cos += np.cos(phase)
            common_sum_sin_n += np.sin(phase) / n

        # dy/da = 0.6366 * sum(sin(phi)/n)
        self.amplitude.backward(
            grad_output * 0.6366 * common_sum_sin_n, context, sample_rate
        )

        # dphi/dt = 2pi * f * n
        # dphi/df = 2pi * t * n
        # dy/dt = 0.6366 * a * sum(cos(phi) * 2pi * f) = 0.6366 * a * 2pi * f * sum(cos(phi))
        grad_base = grad_output * 0.6366 * a_v * self.pi2 * common_sum_cos
        self.time.backward(grad_base * f_v, context, sample_rate)
        self.frequency.backward(grad_base * t_v, context, sample_rate)
