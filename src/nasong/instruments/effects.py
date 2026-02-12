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


"""Instrument-specific effects and utility values.

This module provides specialized envelope generators and value modulators used
by various instruments. Note that some components here are "Value Generators"
rather than traditional audio-input effects.
"""

#
### Import Modules. ###
#
import math

#
import numpy as np
from numpy.typing import NDArray

#
import nasong.core.all_values as lv


#
class ADSR_Piano(lv.Value):
    """Looping Attack-Decay-Sustain-Release envelope generator.

    Warning:
        This class uses modulo arithmetic on time, causing the envelope to loop.
        For one-shot note envelopes, use `ADSR2` instead.

    Attributes:
        time (lv.Value): The global time value.
        note_freq (float): Unused, kept for compatibility.
        attack (float): Attack time in seconds.
        decay (float): Decay time in seconds.
        sustain_level (float): Sustain volume (0.0 to 1.0).
        release (float): Release time in seconds.
        note_duration (float): Duration before release phase begins.
        total_cycle_time (float): Total loop duration (duration + release).
    """

    #
    def __init__(
        self,
        time: lv.Value,
        note_freq: float,
        attack: float = 0.05,
        decay: float = 0.1,
        sustain_level: float = 0.7,
        release: float = 0.3,
        note_duration: float = 1.0,
    ) -> None:
        """Initializes the looping ADSR envelope.

        Args:
            time (lv.Value): Time provider.
            note_freq (float): Target frequency (unused).
            attack (float, optional): Attack duration. Defaults to 0.05.
            decay (float, optional): Decay duration. Defaults to 0.1.
            sustain_level (float, optional): Sustain level. Defaults to 0.7.
            release (float, optional): Release duration. Defaults to 0.3.
            note_duration (float, optional): Gated duration. Defaults to 1.0.
        """

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.note_freq: float = note_freq
        self.attack: float = attack
        self.decay: float = decay
        self.sustain_level: float = sustain_level
        self.release: float = release
        self.note_duration: float = note_duration
        self.total_cycle_time: float = note_duration + release

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Calculates a single sample of the looping envelope.

        Args:
            index (int): Sample index.
            sample_rate (int): Audio sample rate.

        Returns:
            float: The envelope value at the given index.
        """

        #
        t: float = self.time.get_item(index=index, sample_rate=sample_rate)
        #
        t_mod: float = t % self.total_cycle_time

        #
        ### Attack phase. ###
        #
        if t_mod < self.attack:
            #
            return t_mod / self.attack
        #
        ### Decay phase. ###
        #
        elif t_mod < self.attack + self.decay:
            #
            progress: float = (t_mod - self.attack) / self.decay
            #
            return 1.0 - (1.0 - self.sustain_level) * progress
        #
        ### Sustain phase. ###
        #
        elif t_mod < self.note_duration:
            #
            return self.sustain_level
        #
        ### Release phase. ###
        #
        elif t_mod < self.total_cycle_time:
            #
            progress: float = (t_mod - self.note_duration) / self.release
            #
            return self.sustain_level * (1.0 - progress)
        #
        else:
            #
            return 0.0

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Calculates a buffer of envelope samples using NumPy.

        Args:
            indexes_buffer (NDArray[np.float32]): Array of sample indices.
            sample_rate (int): Audio sample rate.

        Returns:
            NDArray[np.float32]: Buffer of envelope values.
        """

        #
        ### Get the time buffer and apply the looping modulo operator. ###
        #
        t: NDArray[np.float32] = self.time.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )
        #
        t_mod: NDArray[np.float32] = np.mod(t, self.total_cycle_time)

        #
        ### Define the 4 stages and their values. ###
        #
        attack_mask: NDArray[np.bool_] = t_mod < self.attack
        attack_val: NDArray[np.float32] = t_mod / self.attack

        #
        decay_mask: NDArray[np.bool_] = t_mod < (self.attack + self.decay)
        decay_progress: NDArray[np.float32] = (t_mod - self.attack) / self.decay
        decay_val: NDArray[np.float32] = (
            1.0 - (1.0 - self.sustain_level) * decay_progress
        ).astype(dtype=np.float32)

        #
        sustain_mask: NDArray[np.bool_] = t_mod < self.note_duration
        sustain_val: NDArray[np.float32] = np.full_like(t_mod, self.sustain_level)

        #
        ### The final 'else' is the release phase. ###
        #
        release_progress: NDArray[np.float32] = (
            (t_mod - self.note_duration) / self.release
        ).astype(dtype=np.float32)
        release_val: NDArray[np.float32] = (
            self.sustain_level * (1.0 - release_progress)
        ).astype(dtype=np.float32)

        #
        ### Build the envelope with nested np.where ###
        ### np.where(condition, if_true, if_false)  ###
        #
        return np.where(
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
class Vibrato(lv.Value):
    """Generates a frequency modulation signal for vibrato.

    This class provides a "truthful" model of Frequency Modulation (FM). It is
    not an audio effect that takes signal input; rather, it is a Value generator
    intended to modulate the `frequency` parameter of an oscillator.

    Attributes:
        time (lv.Value): Global time provider.
        base_frequency (float): The center frequency (Hz).
        vibrato_rate (float): LFO frequency speed (Hz).
        vibrato_depth (float): Modulation width as a percentage of base frequency.
    """

    #
    def __init__(
        self,
        time: lv.Value,
        base_frequency: float,
        vibrato_rate: float = 5.0,  # The LFO frequency in Hz
        vibrato_depth: float = 0.015,  # The modulation amount (e.g., 1.5%)
    ) -> None:
        """
        Initializes the vibrato frequency generator.

        Args:
            time: The global time `lv.Value`.
            base_frequency: The center frequency (in Hz).
            vibrato_rate: The speed of the wobble (LFO freq in Hz).
            vibrato_depth: The "width" of the wobble (as a percentage).
        """

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.base_frequency: float = base_frequency
        self.vibrato_rate: float = vibrato_rate
        self.vibrato_depth: float = vibrato_depth
        self.pi2: float = 2 * math.pi

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Calculates a single sample of the vibrato frequency.

        Args:
            index (int): Sample index.
            sample_rate (int): Audio sample rate.

        Returns:
            float: The modulated frequency value.
        """

        #
        t: float = self.time.get_item(index=index, sample_rate=sample_rate)

        #
        ### Calculate the LFO value (a sine wave oscillating between -1 and 1). ###
        #
        modulation: float = math.sin(self.pi2 * self.vibrato_rate * t)

        #
        ### Apply modulation to the base frequency. ###
        #
        return self.base_frequency * (1.0 + self.vibrato_depth * modulation)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Calculates a buffer of vibrato frequencies using NumPy.

        Args:
            indexes_buffer (NDArray[np.float32]): Array of sample indices.
            sample_rate (int): Audio sample rate.

        Returns:
            NDArray[np.float32]: Buffer of modulated frequency values.
        """

        #
        ### Get the time buffer. ###
        #
        t: NDArray[np.float32] = self.time.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

        #
        ### Calculate the LFO value (a sine wave oscillating between -1 and 1). ###
        #
        modulation: NDArray[np.float32] = np.sin(self.pi2 * self.vibrato_rate * t)

        #
        ### Apply modulation to the base frequency. ###
        #
        return (self.base_frequency * (1.0 + self.vibrato_depth * modulation)).astype(
            dtype=np.float32
        )
