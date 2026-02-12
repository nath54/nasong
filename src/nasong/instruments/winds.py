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


"""Wind instrument definitions.

This module provides synthesizers for wind and reed instruments, such as the
Saxophone, using additive synthesis and noise injection.
"""

#
### Import Modules. ###
#
import math

#
import nasong.core.all_values as lv


#
### CATEGORY: WINDS ###
#


def SaxophoneNote(
    time: lv.Value,
    frequency: float,
    start_time: float,
    duration: float,
    amplitude: float = 0.3,
) -> lv.Value:
    """Simulates a saxophone note.

    Uses an odd-harmonic heavy additive synthesis model to replicate the
    characteristic "reed" buzz, with added vibrato and breath noise.

    Args:
        time (lv.Value): The global time provider.
        frequency (float): The fundamental frequency (Hz).
        start_time (float): activation time in seconds.
        duration (float): Length of the note in seconds.
        amplitude (float, optional): Overall volume scaling. Defaults to 0.3.

    Returns:
        lv.Value: The audio value graph for the saxophone.
    """

    #
    ### Envelope: ASR  ###
    #
    amp_env: lv.Value = lv.ADSR2(
        time,
        note_start=start_time,
        note_duration=duration,
        attack_time=0.1,
        decay_time=0.001,
        sustain_level=1.0,
        release_time=0.15,
    )

    #
    ### Vibrato LFO: 5.5 Hz rate, 0.01 depth  ###
    #
    vibrato_lfo: lv.Value = lv.LFO(
        time,
        rate_hz=lv.c(5.5),
        waveform_class=lv.Sin,
        amplitude=lv.c(0.01 * frequency),  # Depth is relative
    )
    #
    osc_freq: lv.Value = lv.Sum(lv.c(frequency), vibrato_lfo)

    #
    ### Harmonics (odd-heavy) , using `time` (t) ###
    #
    pi2: float = 2 * math.pi
    freq_rad: lv.Value = lv.Product(osc_freq, lv.c(pi2))
    #
    h1: lv.Value = lv.Sin(time, freq_rad, lv.c(1.0))
    h2: lv.Value = lv.Sin(time, lv.Product(freq_rad, lv.c(2)), lv.c(0.3))
    h3: lv.Value = lv.Sin(time, lv.Product(freq_rad, lv.c(3)), lv.c(0.6))
    h4: lv.Value = lv.Sin(time, lv.Product(freq_rad, lv.c(4)), lv.c(0.15))
    h5: lv.Value = lv.Sin(time, lv.Product(freq_rad, lv.c(5)), lv.c(0.4))
    #
    harmonics: lv.Value = lv.Sum(h1, h2, h3, h4, h5)

    #
    ### Breath noise  ###
    #
    breath: lv.Value = lv.WhiteNoise(seed=7919, scale=(1 / 1000.0 * 0.5))

    #
    signal: lv.Value = lv.Sum(harmonics, breath)

    #
    ### Final = Amplitude * AmpEnv * Signal ###
    #
    return lv.Product(lv.c(amplitude), amp_env, signal)
