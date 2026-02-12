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
Utility functions for complex Value operations.

This module provides high-level utilities for generating harmonic series and
creating Low-Frequency Oscillators (LFOs).
"""

#
### Import Modules. ###
#
from typing import Callable

#
import math

#
from nasong.core.value import Value
from nasong.core.values.basic.value_constant import Constant
from nasong.core.values.complex.value_sin import Sin
from nasong.core.values.mult_itms_ops.value_product import Product
from nasong.core.values.mult_itms_ops.value_sum import Sum

#
### UTILS. ###
#


#
def generate_harmonics(
    time: Value,
    base_frequency: float,
    num_harmonics: int,
    amplitude_falloff: float,
    sample_rate: int,
    base_amplitude: Value = Constant(1.0),
) -> Value:
    """Utility to generate a band-limited sum of harmonic sine waves.

    This function creates a series of sine waves starting at the base frequency
    and adding harmonics (multiples of the base frequency). It prevents
    aliasing by only adding harmonics that are below the Nyquist frequency.

    Args:
        time (Value): The time input for the sine oscillators.
        base_frequency (float): The fundamental frequency in Hz (e.g., 440.0).
        num_harmonics (int): The maximum number of harmonics to generate.
        amplitude_falloff (float): A multiplier for each successive harmonic's
            amplitude (e.g., 0.5 means each harmonic is half the amplitude of
            the previous one).
        sample_rate (int): The audio sample rate in Hz (e.g., 44100).
        base_amplitude (Value, optional): Overall amplitude scaling.
            Defaults to 1.0.

    Returns:
        Value: A Sum object containing all valid, band-limited harmonics.
    """

    #
    harmonics_list: list[Value] = []
    nyquist_limit: float = sample_rate / 2.0
    pi2: float = 2 * math.pi
    current_amplitude_multiplier: float = 1.0

    #
    for n in range(1, num_harmonics + 1):
        #
        ### Calculate the frequency of the Nth harmonic. ###
        #
        harmonic_freq_hz: float = base_frequency * n

        #
        ### This is the anti-aliasing check. ###
        #
        if harmonic_freq_hz >= nyquist_limit:
            #
            break  # Stop adding harmonics that are too high.

        #
        ### Calculate the amplitude for this harmonic. ###
        #
        amp_value: Value = Product(
            base_amplitude, Constant(current_amplitude_multiplier)
        )

        #
        ### Add the new Sin wave to our list. ###
        #
        harmonics_list.append(
            Sin(
                value=time,
                frequency=Constant(harmonic_freq_hz * pi2),
                amplitude=amp_value,
            )
        )

        #
        ### Apply the falloff for the *next* harmonic. ###
        #
        current_amplitude_multiplier *= amplitude_falloff

    #
    ### If no harmonics were valid, return silence. ###
    #
    if not harmonics_list:
        #
        return Constant(0.0)

    #
    ### Return a single Value object that sums all harmonics. ###
    #
    return Sum(harmonics_list)


#
def LFO(
    time: Value,
    rate_hz: Value,
    waveform_class: Callable[..., Value],
    amplitude: Value = Constant(1.0),
    delta: Value = Constant(0.0),
) -> Value:
    """Utility to create a Low-Frequency Oscillator (LFO).

    This helper function simplifies LFO creation by abstracting unit
    conversions for different oscillator types. It ensures that `rate_hz`
    is always treated as Hertz, converting it to radians per second where
    required (e.g., for `Sin` and `Cos`).

    Args:
        time (Value): The time input for the oscillator.
        rate_hz (Value): The LFO frequency in Hz.
        waveform_class (Callable[..., Value]): The Value class to instantiate
            (e.g., Sin, Triangle, Square).
        amplitude (Value, optional): The LFO amplitude. Defaults to 1.0.
        delta (Value, optional): The initial phase/offset. Defaults to 0.0.

    Returns:
        Value: An instance of the requested waveform class configured as an LFO.
    """

    #
    freq_val: Value

    #
    ### Check if the oscillator is Sin/Cos, which need rad/s. ###
    #
    if waveform_class.__name__ == "Sin" or waveform_class.__name__ == "Cos":
        #
        ### Convert Hz to rad/s (Hz * 2 * pi). ###
        #
        freq_val = Product(rate_hz, Constant(2 * math.pi))
    #
    else:
        #
        ### Triangle, Square, etc., already use Hz. ###
        #
        freq_val = rate_hz

    #
    ### Return the instantiated oscillator class. ###
    #
    return waveform_class(
        value=time, frequency=freq_val, amplitude=amplitude, delta=delta
    )
