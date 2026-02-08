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
    """
    Utility to generate a band-limited sum of harmonic sine waves.

    This function is crucial for "good listening" as it prevents
    aliasing by checking harmonics against the Nyquist frequency.

    Args:
        time: The base `Value` for time (e.g., from `song.render`).
        base_frequency: The fundamental frequency in Hz (e.g., 440.0).
        num_harmonics: The maximum number of harmonics to generate.
        amplitude_falloff: A multiplier for each successive harmonic's
            amplitude. (e.g., 0.5 means each harmonic is half the
            amplitude of the previous one).
        sample_rate: The audio sample rate (e.g., 44100).
        base_amplitude: A `Value` for the fundamental's amplitude.

    Returns:
        A `Sum` Value object containing all valid, band-limited harmonics.
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
    """
    Utility to create a Low-Frequency Oscillator (LFO).

    This helper function simplifies LFO creation by abstracting the
    frequency unit inconsistency in the oscillator APIs.
    - `Sin` and `Cos` expect frequency in radians per second.

    - `Triangle`, `Square`, `Sawtooth` expect frequency in Hz.


    This function always takes `rate_hz` in Hz and automatically
    converts it to the correct unit for the given `waveform_class`.
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
