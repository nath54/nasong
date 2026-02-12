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
ADSR envelope implementation (v2).

This module provides the `ADSR2` class, which implements a one-shot
Attack-Decay-Sustain-Release (ADSR) envelope. It is designed to be "truthful"
to the standard ADSR model while supporting vectorized operations and
differentiability.

Example:
    >>> from nasong.core.values.basic.value_identity import Identity
    >>> from nasong.core.values.complex.value_adsr2 import ADSR2
    >>> time = Identity()
    >>> adsr = ADSR2(time=time, note_start=0, note_duration=1.0)
    >>> adsr.get_item(0, 44100)
    0.0
"""

#
### Import Modules. ###
#
from typing import Any
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant


#
class ADSR2(Value):
    """A "truthful" one-shot Attack-Decay-Sustain-Release envelope.

    This class generates a standard ADSR volume envelope based on a trigger
    time and duration. It is "one-shot" in the sense that it follows the full
    ADSR sequence once triggered.

    Attributes:
        time (Value): The time source (usually an Identity or similar).
        note_start (Value): The time when the note starts (Attack trigger).
        note_duration (Value): The duration of the note (Sustain duration).
        attack_time (Value): The time it takes to reach peak level.
        decay_time (Value): The time it takes to settle to sustain level.
        sustain_level (Value): The level maintained after decay until release.
        release_time (Value): The time it takes to reach zero after note ends.
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
        """Initializes the ADSR2 envelope.

        Args:
            time (Value): The time source.
            note_start (Value | float): Start time of the envelope.
            note_duration (Value | float): Duration of the sustain phase.
            attack_time (Value | float): Duration of the attack phase.
            decay_time (Value | float): Duration of the decay phase.
            sustain_level (Value | float): Amplitude level during sustain.
            release_time (Value | float): Duration of the release phase.
        """

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
        """Returns the envelope value for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The envelope amplitude at the given index.
        """
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
        """Returns a vectorized NumPy array of the envelope.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized envelope samples.
        """
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
        """Generates the envelope for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of envelope samples.
        """

        #
        t: Tensor = self.time.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        #
        # Fetch current values for all parameters
        #
        start: Tensor = self.note_start.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        dur: Tensor = self.note_duration.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        att: Tensor = self.attack_time.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        ).clamp(min=1e-6)
        dec: Tensor = self.decay_time.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        ).clamp(min=1e-6)
        sus: Tensor = self.sustain_level.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )
        rel: Tensor = self.release_time.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        ).clamp(min=1e-6)

        #
        relative_time: Tensor = t - start

        #
        ### Gate: Create a mask for all samples inside the envelope's lifetime. ###
        #
        release_end = dur + rel
        gate_mask: Tensor = ((relative_time >= 0) & (relative_time <= release_end)).to(
            dtype=torch.float32
        )

        #
        if not torch.any(gate_mask):
            #
            return torch.zeros_like(indexes_buffer, dtype=torch.float32, device=device)

        #
        ### Define the 4 stages and their values. ###
        #
        att_end = att
        dec_end = att + dec
        sus_end = dur

        attack_mask: Tensor = relative_time < att_end
        attack_val: Tensor = relative_time / att

        #
        decay_mask: Tensor = (relative_time >= att_end) & (relative_time < dec_end)
        decay_progress: Tensor = (relative_time - att) / dec
        decay_val: Tensor = (1.0 - (1.0 - sus) * decay_progress).to(dtype=torch.float32)

        #
        sustain_mask: Tensor = (relative_time >= dec_end) & (relative_time < sus_end)
        sustain_val: Tensor = sus

        #
        ### The final 'else' is the release phase. ###
        #
        release_progress: Tensor = (relative_time - dur) / rel
        release_val: Tensor = (sus * (1.0 - release_progress)).to(dtype=torch.float32)

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
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the ADSR envelope.

        The gradient dy/dt is computed piecewise for each ADSR stage:
        - Attack: 1/attack_time
        - Decay: -(1-sustain_level)/decay_time
        - Sustain: 0
        - Release: -sustain_level/release_time

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
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
        start = self.note_start.getitem_np(
            np.zeros(grad_output.shape, dtype=np.float32), sample_rate
        )
        dur = self.note_duration.getitem_np(
            np.zeros(grad_output.shape, dtype=np.float32), sample_rate
        )
        att = np.clip(
            self.attack_time.getitem_np(
                np.zeros(grad_output.shape, dtype=np.float32), sample_rate
            ),
            1e-6,
            None,
        )
        dec = np.clip(
            self.decay_time.getitem_np(
                np.zeros(grad_output.shape, dtype=np.float32), sample_rate
            ),
            1e-6,
            None,
        )
        sus = self.sustain_level.getitem_np(
            np.zeros(grad_output.shape, dtype=np.float32), sample_rate
        )
        rel = np.clip(
            self.release_time.getitem_np(
                np.zeros(grad_output.shape, dtype=np.float32), sample_rate
            ),
            1e-6,
            None,
        )

        relative_time = t - start
        att_end = att
        dec_end = att + dec
        sus_end = dur
        release_end = dur + rel

        dy_dt: NDArray[np.float32] = np.zeros_like(relative_time)

        # Stage masks
        gate_mask = (relative_time >= 0) & (relative_time <= release_end)
        attack_mask = relative_time < att_end
        decay_mask = (relative_time >= att_end) & (relative_time < dec_end)
        sustain_mask = (relative_time >= dec_end) & (relative_time < sus_end)
        release_mask = (relative_time >= sus_end) & (relative_time <= release_end)

        dy_dt[attack_mask] = 1.0 / att[attack_mask]
        dy_dt[decay_mask] = -(1.0 - sus[decay_mask]) / dec[decay_mask]
        dy_dt[sustain_mask] = 0.0
        dy_dt[release_mask] = -sus[release_mask] / rel[release_mask]

        self.time.backward(grad_output * dy_dt * gate_mask, context, sample_rate)
