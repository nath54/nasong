from typing import Dict, Any
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor


#
class ADSR2(Value):
    """
    A "truthful" one-shot Attack-Decay-Sustain-Release envelope.
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
    ) -> None:

        #
        super().__init__()

        #
        self.time: Value = time
        self.note_start: float = note_start
        self.note_duration: float = note_duration
        self.attack_time: float = attack_time
        self.decay_time: float = decay_time
        self.sustain_level: float = sustain_level
        self.release_time: float = release_time

        #
        ### Pre-calculate stage end times for clarity. ###
        #
        self.attack_end: float = self.attack_time
        self.decay_end: float = self.attack_time + self.decay_time
        self.sustain_end: float = self.note_duration  # This is the "note off" event
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
            return relative_time / self.attack_time

        #
        ### Decay phase. ###
        #
        elif relative_time < self.decay_end:
            #
            decay_progress: float = (relative_time - self.attack_time) / self.decay_time
            #
            return 1.0 - (1.0 - self.sustain_level) * decay_progress

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
            release_progress: float = (
                relative_time - self.note_duration
            ) / self.release_time
            #
            return self.sustain_level * (1.0 - release_progress)

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
        attack_mask: NDArray[np.bool_] = relative_time < self.attack_end
        attack_val: NDArray[np.float32] = relative_time / self.attack_time

        #
        decay_mask: NDArray[np.bool_] = relative_time < self.decay_end
        decay_progress: NDArray[np.float32] = (
            relative_time - self.attack_time
        ) / self.decay_time
        decay_val: NDArray[np.float32] = (
            1.0 - (1.0 - self.sustain_level) * decay_progress
        ).astype(dtype=np.float32)

        #
        sustain_mask: NDArray[np.bool_] = relative_time < self.sustain_end
        sustain_val: NDArray[np.float32] = np.full_like(
            relative_time, self.sustain_level
        )

        #
        ### The final 'else' is the release phase. ###
        #
        release_progress: NDArray[np.float32] = (
            relative_time - self.note_duration
        ) / self.release_time
        release_val: NDArray[np.float32] = (
            self.sustain_level * (1.0 - release_progress)
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
        attack_mask: Tensor = relative_time < self.attack_end
        attack_val: Tensor = relative_time / self.attack_time

        #
        decay_mask: Tensor = relative_time < self.decay_end
        decay_progress: Tensor = (relative_time - self.attack_time) / self.decay_time
        decay_val: Tensor = (1.0 - (1.0 - self.sustain_level) * decay_progress).to(
            dtype=torch.float32
        )

        #
        sustain_mask: Tensor = relative_time < self.sustain_end
        sustain_val: Tensor = torch.full_like(
            relative_time, self.sustain_level, device=device
        )

        #
        ### The final 'else' is the release phase. ###
        #
        release_progress: Tensor = (
            relative_time - self.note_duration
        ) / self.release_time
        release_val: Tensor = (self.sustain_level * (1.0 - release_progress)).to(
            dtype=torch.float32
        )

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

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradient to self.time.
        dy/dt is piecewise:
        - Attack: 1/attack_time
        - Decay: -(1-sustain_level)/decay_time
        - Sustain: 0
        - Release: -sustain_level/release_time
        """
        # We need the relative time again.
        # For performance, we could save it in getitem_np,
        # but recalculating it from time's forward result is also possible.
        # However, to be strict, we'd need time.getitem_np's result.
        # Let's assume for now we recalculate for simplicity,
        # or we should have saved it.
        t: NDArray[np.float32] = self.time.getitem_np(
            np.zeros(grad_output.shape, dtype=np.float32), sample_rate
        )
        relative_time: NDArray[np.float32] = t - self.note_start

        dy_dt: NDArray[np.float32] = np.zeros_like(relative_time)

        # Stage masks
        gate_mask = (relative_time >= 0) & (relative_time <= self.release_end)
        attack_mask = relative_time < self.attack_end
        decay_mask = (relative_time >= self.attack_end) & (
            relative_time < self.decay_end
        )
        sustain_mask = (relative_time >= self.decay_end) & (
            relative_time < self.sustain_end
        )
        release_mask = (relative_time >= self.sustain_end) & (
            relative_time <= self.release_end
        )

        dy_dt[attack_mask] = 1.0 / self.attack_time
        dy_dt[decay_mask] = -(1.0 - self.sustain_level) / self.decay_time
        dy_dt[sustain_mask] = 0.0
        dy_dt[release_mask] = -self.sustain_level / self.release_time

        self.time.backward(grad_output * dy_dt * gate_mask, context, sample_rate)
