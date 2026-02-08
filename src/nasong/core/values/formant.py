#
### Import Modules. ###
#

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
    """
    Data structure for a formant resonance.
    """

    #
    def __init__(self, freq: float, gain_db: float, q: float) -> None:

        #
        self.freq: float = freq
        self.gain: float = 10.0 ** (gain_db / 20.0)
        self.q: float = q

    #
    def __repr__(self) -> str:

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
    """
    Generates harmonics shaped by a set of formants (resonances).
    This simulates the body resonance of an instrument (like a violin or cello).
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
