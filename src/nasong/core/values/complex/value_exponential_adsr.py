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
class ExponentialADSR(Value):
    """
    A "truthful" one-shot Attack-Decay-Sustain-Release envelope
    with *exponential* curves.

    This is an improvement on ADSR2, allowing for non-linear (concave or
    convex) curves for the Attack, Decay, and Release stages.

    - A curve value of 1.0 is linear (identical to ADSR2).
    - A curve value > 1.0 is "convex" (starts slow, ends fast).
    - A curve value < 1.0 is "concave" (starts fast, ends slow).
    """

    #
    def __init__(
        self,
        time: Value,
        note_start: float,
        note_duration: float,
        attack_time: float = 0.05,
        decay_time: float = 0.1,
        sustain_level: float = 0.7,
        release_time: float = 0.2,
        attack_curve: float = 0.5,  # Concave (fast) start
        decay_curve: float = 2.0,  # Convex (natural) decay
        release_curve: float = 2.0,  # Convex (natural) release
    ) -> None:

        #
        super().__init__()

        #
        self.time: Value = time
        self.note_start: float = note_start
        self.note_duration: float = note_duration
        self.attack_time: float = max(0.001, attack_time)  # Prevent div by zero
        self.decay_time: float = max(0.001, decay_time)
        self.sustain_level: float = sustain_level
        self.release_time: float = max(0.001, release_time)

        #
        self.attack_curve: float = attack_curve
        self.decay_curve: float = decay_curve
        self.release_curve: float = release_curve

        #
        ### Pre-calculate stage end times. ###
        #
        self.attack_end: float = self.attack_time
        self.decay_end: float = self.attack_time + self.decay_time
        self.sustain_end: float = self.note_duration  # "note off" event
        self.release_end: float = self.note_duration + self.release_time

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.get_item(index=index, sample_rate=sample_rate)
        relative_time: float = t - self.note_start

        #
        ### Gate: if we are before the note or after the release, output 0. ###
        #
        if relative_time < 0 or relative_time > self.release_end:
            #
            return 0.0

        #
        ### Attack phase. ###
        #
        if relative_time < self.attack_end:
            #
            progress: float = relative_time / self.attack_time
            #
            return math.pow(progress, self.attack_curve)

        #
        ### Decay phase. ###
        #
        elif relative_time < self.decay_end:
            #
            progress: float = (relative_time - self.attack_time) / self.decay_time
            #
            return self.sustain_level + (1.0 - self.sustain_level) * math.pow(
                1.0 - progress, self.decay_curve
            )

        #
        ### Sustain phase. ###
        #
        elif relative_time < self.sustain_end:
            #
            return self.sustain_level

        #
        ### Release phase. ###
        #
        else:
            #
            progress: float = (relative_time - self.note_duration) / self.release_time
            #
            return self.sustain_level * math.pow(1.0 - progress, self.release_curve)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        t: NDArray[np.float32] = self.time.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        #
        relative_time: NDArray[np.float32] = t - self.note_start

        #
        ### Gate: Create a mask for all samples inside the envelope's lifetime. ###
        #
        gate_mask: NDArray[np.bool_] = (relative_time >= 0) & (
            relative_time <= self.release_end
        )

        #
        if not np.any(gate_mask):
            #
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        ### Define the 4 stages and their values. ###
        #

        #
        ## ATTACK. ##
        #
        attack_mask: NDArray[np.bool_] = relative_time < self.attack_end
        attack_progress: NDArray[np.float32] = (
            relative_time / self.attack_time
        ).astype(dtype=np.float32)
        #
        ## Ensure base is not negative (which happens if relative_time < 0). ##
        #
        attack_base = np.maximum(0.0, attack_progress)
        attack_val: NDArray[np.float32] = np.power(attack_base, self.attack_curve)

        #
        ## DECAY. ##
        #
        decay_mask: NDArray[np.bool_] = relative_time < self.decay_end
        decay_progress: NDArray[np.float32] = (
            (relative_time - self.attack_time) / self.decay_time
        ).astype(dtype=np.float32)
        #
        ## Ensure base is not negative (which happens if progress > 1.0). ##
        #
        decay_base = np.maximum(0.0, 1.0 - decay_progress)
        decay_val: NDArray[np.float32] = (
            self.sustain_level
            + (1.0 - self.sustain_level) * np.power(decay_base, self.decay_curve)
        ).astype(dtype=np.float32)

        #
        ## SUSTAIN. ##
        #
        sustain_mask: NDArray[np.bool_] = relative_time < self.sustain_end
        sustain_val: NDArray[np.float32] = np.full_like(
            relative_time, self.sustain_level
        )

        #
        ## RELEASE. ##
        #
        release_progress: NDArray[np.float32] = (
            (relative_time - self.note_duration) / self.release_time
        ).astype(dtype=np.float32)
        #
        ## Ensure base is not negative. ##
        #
        release_base = np.maximum(0.0, 1.0 - release_progress)
        release_val: NDArray[np.float32] = (
            self.sustain_level * np.power(release_base, self.release_curve)
        ).astype(dtype=np.float32)

        #
        ### Build the envelope with nested np.where. ###
        #
        env: NDArray[np.float32] = np.where(
            attack_mask,
            attack_val,
            np.where(
                decay_mask,
                decay_val,
                np.where(
                    sustain_mask,
                    sustain_val,
                    release_val,  # The final 'else' case
                ),
            ),
        )

        #
        ### Apply the main gate mask to ensure output is 0 outside the envelope. ###
        #
        return env * gate_mask

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

        #
        t: Tensor = self.time.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        #
        relative_time: Tensor = t - self.note_start

        #
        ### Gate: Create a mask for all samples inside the envelope's lifetime. ###
        #
        gate_mask: Tensor = (
            (relative_time >= 0) & (relative_time <= self.release_end)
        ).to(dtype=torch.float32)

        #
        if not torch.any(gate_mask):
            #
            return torch.zeros_like(indexes_buffer, dtype=torch.float32, device=device)

        #
        ### Define the 4 stages and their values. ###
        #

        #
        ## ATTACK. ##
        #
        attack_mask: Tensor = relative_time < self.attack_end
        attack_progress: Tensor = (relative_time / self.attack_time).to(
            dtype=torch.float32
        )
        #
        ## Ensure base is not negative (which happens if relative_time < 0). ##
        #
        attack_base = torch.maximum(
            torch.tensor(0.0, dtype=torch.float32, device=device), attack_progress
        )
        attack_val: Tensor = torch.pow(attack_base, self.attack_curve)

        #
        ## DECAY. ##
        #
        decay_mask: Tensor = relative_time < self.decay_end
        decay_progress: Tensor = (
            (relative_time - self.attack_time) / self.decay_time
        ).to(dtype=torch.float32)
        #
        ## Ensure base is not negative (which happens if progress > 1.0). ##
        #
        decay_base = torch.maximum(
            torch.tensor(0.0, dtype=torch.float32, device=device), 1.0 - decay_progress
        )
        decay_val: Tensor = (
            self.sustain_level
            + (1.0 - self.sustain_level) * torch.pow(decay_base, self.decay_curve)
        ).to(dtype=torch.float32)

        #
        ## SUSTAIN. ##
        #
        sustain_mask: Tensor = relative_time < self.sustain_end
        sustain_val: Tensor = torch.full_like(
            relative_time, self.sustain_level, device=device
        )

        #
        ## RELEASE. ##
        #
        release_progress: Tensor = (
            (relative_time - self.note_duration) / self.release_time
        ).to(dtype=torch.float32)
        #
        ## Ensure base is not negative. ##
        #
        release_base = torch.maximum(
            torch.tensor(0.0, dtype=torch.float32, device=device),
            1.0 - release_progress,
        )
        release_val: Tensor = (
            self.sustain_level * torch.pow(release_base, self.release_curve)
        ).to(dtype=torch.float32)

        #
        ### Build the envelope with nested torch.where. ###
        #
        env: Tensor = torch.where(
            attack_mask,
            attack_val,
            torch.where(
                decay_mask,
                decay_val,
                torch.where(
                    sustain_mask,
                    sustain_val,
                    release_val,  # The final 'else' case
                ),
            ),
        )

        #
        ### Apply the main gate mask to ensure output is 0 outside the envelope. ###
        #
        return env * gate_mask
