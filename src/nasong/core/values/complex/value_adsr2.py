# Copyright (C) 2026 Nathan Cerisara <https://github.com/nath54/nasong>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""
TODO: add full docstring, explaining what the goal of this script is, and explaining for each class and each function what is it, how it works, and how to use it.
"""

#
### Import Modules. ###
#
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
        note_start: Value | float,
        note_duration: Value | float,
        attack_time: Value | float = 0.05,
        decay_time: Value | float = 0.1,
        sustain_level: Value | float = 0.7,
        release_time: Value | float = 0.2,
    ) -> None:
        from nasong.core.values.basic.value_constant import Constant

        #
        super().__init__()

        #
        self.time: Value = time

        def wrap(v):
            return v if isinstance(v, Value) else Constant(v)

        self.note_start: Value = wrap(note_start)
        self.note_duration: Value = wrap(note_duration)
        self.attack_time: Value = wrap(attack_time)
        self.decay_time: Value = wrap(decay_time)
        self.sustain_level: Value = wrap(sustain_level)
        self.release_time: Value = wrap(release_time)

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        t: float = self.time.get_item(index=index, sample_rate=sample_rate)
        start: float = self.note_start.get_item(index=index, sample_rate=sample_rate)
        dur: float = self.note_duration.get_item(index=index, sample_rate=sample_rate)
        att: float = max(
            self.attack_time.get_item(index=index, sample_rate=sample_rate), 1e-6
        )
        dec: float = max(
            self.decay_time.get_item(index=index, sample_rate=sample_rate), 1e-6
        )
        sus: float = self.sustain_level.get_item(index=index, sample_rate=sample_rate)
        rel: float = max(
            self.release_time.get_item(index=index, sample_rate=sample_rate), 1e-6
        )

        relative_time: float = t - start
        release_end = dur + rel

        if relative_time < 0 or relative_time > release_end:
            return 0.0

        att_end = att
        dec_end = att + dec
        sus_end = dur

        if relative_time < att_end:
            return relative_time / att
        elif relative_time < dec_end:
            decay_progress = (relative_time - att) / dec
            return 1.0 - (1.0 - sus) * decay_progress
        elif relative_time < sus_end:
            return sus
        else:
            release_progress = (relative_time - dur) / rel
            return sus * (1.0 - release_progress)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        t = self.time.getitem_np(indexes_buffer, sample_rate)
        start = self.note_start.getitem_np(indexes_buffer, sample_rate)
        dur = self.note_duration.getitem_np(indexes_buffer, sample_rate)
        att = np.clip(
            self.attack_time.getitem_np(indexes_buffer, sample_rate), 1e-6, None
        )
        dec = np.clip(
            self.decay_time.getitem_np(indexes_buffer, sample_rate), 1e-6, None
        )
        sus = self.sustain_level.getitem_np(indexes_buffer, sample_rate)
        rel = np.clip(
            self.release_time.getitem_np(indexes_buffer, sample_rate), 1e-6, None
        )

        relative_time = t - start
        release_end = dur + rel

        gate_mask = (relative_time >= 0) & (relative_time <= release_end)
        if not np.any(gate_mask):
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        att_end = att
        dec_end = att + dec
        sus_end = dur

        attack_mask = relative_time < att_end
        attack_val = relative_time / att

        decay_mask = (relative_time >= att_end) & (relative_time < dec_end)
        decay_progress = (relative_time - att) / dec
        decay_val = (1.0 - (1.0 - sus) * decay_progress).astype(np.float32)

        sustain_mask = (relative_time >= dec_end) & (relative_time < sus_end)
        sustain_val = sus

        release_mask = relative_time >= sus_end
        release_progress = (relative_time - dur) / rel
        release_val = (sus * (1.0 - release_progress)).astype(np.float32)

        env = np.zeros_like(relative_time, dtype=np.float32)
        env = np.where(attack_mask, attack_val, env)
        env = np.where(decay_mask, decay_val, env)
        env = np.where(sustain_mask, sustain_val, env)
        env = np.where(release_mask, release_val, env)

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
