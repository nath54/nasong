#
### Import Modules. ###
#
import math

#
import nasong.core.value as lv


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
    """
    Refactored WobbleBass.
    This class was "EXCELLENT"  and is now built
    compositionally from its core components.
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
    distorted: lv.Value = lv.Distortion(filtered, drive=2.0)

    #
    ### Final = Amplitude * Gate * DistortedSignal ###
    #
    return lv.Product(lv.c(amplitude), gate_env, distorted)


#
def DeepBass(
    time: lv.Value, frequency: float, start_time: float, duration: float = 0.5
) -> lv.Value:
    """
    Refactored DeepBass.
    This class was "EXCELLENT"  and is just
    converted to a compositional factory function.
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
    ### Final = 0.4 * Gate * AmpEnv * Signal ###
    #
    return lv.Product(lv.c(0.4), gate_env, amp_env, signal)
