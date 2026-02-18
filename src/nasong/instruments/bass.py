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


"""Bass instrument definitions.

This module provides factory functions for various bass synthesizers, including
electronic wobble bass and deep atmospheric bass sounds. These are built
compositionally using core `Value` components.
"""

#
### Import Modules. ###
#
import math

#
import nasong.core.all_values as lv


#
### CATEGORY: BASS / ELECTRONIC ###
#


def WobbleBass(
    time: lv.Value,
    base_frequency: float,
    start_time: float,
    duration: float,
    wobble_rate: float = 4.0,
    amplitude: float = 0.4,
) -> lv.Value:
    """Creates a classic electronic "wobble" bass.

    This instrument uses a band-limited sawtooth oscillator fed through a
    modulating gain stage (simulating a filter wobble) controlled by an LFO,
    followed by a tanh distortion for extra grit.

    Args:
        time (lv.Value): The global time value.
        base_frequency (float): The fundamental frequency of the bass note.
        start_time (float): The activation time in seconds.
        duration (float): The length of the note in seconds.
        wobble_rate (float, optional): The LFO speed in Hz. Defaults to 4.0.
        amplitude (float, optional): Overall volume scaling. Defaults to 0.4.

    Returns:
        lv.Value: The composite audio value graph for the wobble bass.
    """

    #
    ### Gate: hard gate at duration. ###
    #
    gate_env: lv.Value = lv.ADSR2(time, start_time, duration, 0.001, 0.001, 1.0, 0.001)

    #
    ### LFO (0 to 1 range): (sin(...) + 1) / 2  ###
    #
    lfo_base: lv.Value = lv.LFO(time, lv.c(wobble_rate), lv.Sin)
    #
    lfo_0_to_1: lv.Value = lv.BasicScaling(lfo_base, lv.c(0.5), lv.c(0.5))

    #
    ### Oscillator: 7-harmonic sawtooth  ###
    #
    osc: lv.Value = lv.BandLimitedSawtooth(time, lv.c(base_frequency), num_harmonics=7)

    #
    ### "Filter": osc * (0.3 + 0.7 * lfo)  ###
    #
    filter_mod: lv.Value = lv.BasicScaling(lfo_0_to_1, lv.c(0.7), lv.c(0.3))
    #
    filtered: lv.Value = lv.Product(osc, filter_mod)

    #
    ### Distortion: tanh(filtered * 2.0)  ###
    #
    distorted: lv.Value = lv.Distortion(filtered, gain=2.0)

    #
    ### Final = Amplitude * Gate * DistortedSignal ###
    #
    return lv.Product(lv.c(amplitude), gate_env, distorted)


#
def DeepBass(
    time: lv.Value,
    frequency: float,
    start_time: float,
    duration: float = 0.5,
    amplitude: float = 0.4,
) -> lv.Value:
    """Creates a deep, atmospheric subtractive-style bass note.

    Built using a pure sine wave with an exponential decay envelope, gated
    to a specific duration for clear note boundaries.

    Args:
        time (lv.Value): The global time value.
        frequency (float): The frequency of the bass note.
        start_time (float): The activation time in seconds.
        duration (float, optional): The note duration in seconds. Defaults to 0.5.
        amplitude (float, optional): Overall volume scaling. Defaults to 0.4.

    Returns:
        lv.Value: The audio value graph for the deep bass.
    """

    #
    ### Gate: hard gate at duration. ###
    #
    gate_env: lv.Value = lv.ADSR2(time, start_time, duration, 0.001, 0.001, 1.0, 0.001)

    #
    ### Amplitude envelope: exp(-relative_t * 6)  ###
    #
    amp_env: lv.Value = lv.ExponentialDecay(time, start_time, 6.0)

    #
    ### Oscillator: Pure sine wave  ###
    #
    relative_time: lv.Value = lv.BasicScaling(time, lv.c(1), lv.c(-start_time))
    #
    signal: lv.Value = lv.Sin(relative_time, frequency=lv.c(frequency * 2 * math.pi))

    #
    ### Final = amplitude * Gate * AmpEnv * Signal ###
    #
    return lv.Product(lv.c(amplitude), gate_env, amp_env, signal)
