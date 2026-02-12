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
Formant synthesis implementation.

This module provides tools for formant-based synthesis, which simulates the
resonant characteristics of acoustic instruments or human speech. It includes
the `Formant` data structure and the `generate_formant_harmonics` utility.

Example:
    >>> from nasong.core.values.basic.value_constant import Constant
    >>> from nasong.core.values.formant import Formant, generate_formant_harmonics
    >>> t = Constant(0.0)
    >>> f1 = Formant(freq=500.0, gain_db=0.0, q=10.0)
    >>> v = generate_formant_harmonics(t, 100.0, [f1], 10, 44100)
"""

#
### Import Modules. ###
#
import random
import math

#
from nasong.core.value import Value
from nasong.core.values.basic.value_constant import Constant
from nasong.core.values.complex.value_sin import Sin
from nasong.core.values.mult_itms_ops.value_product import Product
from nasong.core.values.mult_itms_ops.value_sum import Sum

#
### SYSTEM: FORMANT SYNTHESIS ###
#


#
class Formant:
    """Data structure representing a formant resonance.

    A formant is a spectral peak of the spectrum of the voice or an instrument,
    caused by resonance.

    Attributes:
        freq (float): The center frequency of the resonance in Hz.
        gain (float): The linear gain multiplier derived from gain_db.
        q (float): The quality factor (Q) of the resonance, determining its bandwidth.
    """

    #
    def __init__(self, freq: float, gain_db: float, q: float) -> None:
        """Initializes a Formant.

        Args:
            freq (float): Center frequency in Hz.
            gain_db (float): Peak gain in decibels.
            q (float): Quality factor.
        """

        #
        self.freq: float = freq
        self.gain: float = 10.0 ** (gain_db / 20.0)
        self.q: float = q

    #
    def __repr__(self) -> str:
        """Returns a string representation of the formant."""

        #
        return f"Formant(freq={self.freq}, gain={self.gain}, q={self.q})"


#
def generate_formant_harmonics(
    time: Value,
    base_frequency: float,
    formants: list[Formant],
    num_harmonics: int,
    sample_rate: int,
    base_amplitude: Value = Constant(1.0),
    phase_shift: bool = True,
) -> Value:
    """Generates harmonics shaped by a set of formants (resonances).

    This simulates the body resonance of an instrument (like a violin or cello)
    or vocal tract filtering. It creates a series of sine waves at harmonic
    intervals and scales their amplitudes according to the resonant response
    of the provided formants.

    Args:
        time (Value): The time input for the sine oscillators.
        base_frequency (float): The fundamental frequency in Hz.
        formants (list[Formant]): A list of Formant objects defining the spectrum.
        num_harmonics (int): Maximum number of harmonics to generate.
        sample_rate (int): The audio sample rate for anti-aliasing.
        base_amplitude (Value, optional): Overall amplitude scaling. Defaults to 1.0.
        phase_shift (bool, optional): If True, applies random phase shifts to
            harmonics to avoid coherent transients. Defaults to True.

    Returns:
        Value: A Sum of Sin waves representing the formant-shaped harmonics.
    """

    #
    harmonics_list: list[Value] = []
    nyquist_limit: float = sample_rate / 2.0
    pi2: float = 2 * math.pi

    #
    for n in range(1, num_harmonics + 1):
        #
        ### Calculate harmonic frequency. ###
        #
        harmonic_freq_hz: float = base_frequency * n

        #
        ### Anti-aliasing check. ###
        #
        if harmonic_freq_hz >= nyquist_limit:
            break

        #
        ### Calculate amplitude based on formants. ###
        ### We sum the response of all formants at this frequency. ###
        #
        combined_gain: float = 0.0

        #
        for f in formants:
            #
            ### Standard resonant filter response curve calculation. ###
            ### gain = 1 / sqrt( (1 - (f/f_c)^2)^2 + (f/(f_c*Q))^2 ) ###
            #
            f_ratio = harmonic_freq_hz / f.freq
            denom = math.sqrt((1 - f_ratio**2) ** 2 + (f_ratio / f.q) ** 2)
            #
            if denom > 0:
                combined_gain += f.gain / denom

        #
        ### Normalize/Scale gain to avoid clipping if many formants overlap. ###
        #
        # combined_gain = min(combined_gain, 2.0)

        #
        ### Apply 1/n sawtooth falloff as the "source" excitation. ###
        #
        source_amp: float = 1.0 / n
        final_amp: float = source_amp * combined_gain

        #
        ### Create the sine wave. ###
        #
        # Random phase shift to avoid "laser zap" initial transient if requested
        delta: Value = Constant(0.0)
        #
        if phase_shift:
            #
            delta = Constant(random.uniform(0, pi2))

        #
        harmonics_list.append(
            Sin(
                value=time,
                frequency=Constant(harmonic_freq_hz * pi2),
                amplitude=Product(base_amplitude, Constant(final_amp)),
                delta=delta,
            )
        )

    #
    if not harmonics_list:
        #
        return Constant(0.0)

    #
    return Sum(harmonics_list)
