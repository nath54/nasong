#
### Import Modules. ###
#
import math

#
import nasong.core.all_values as lv


#
### CATEGORY: KEYBOARDS ###
#


def PianoNote(
    time: lv.Value,
    frequency: float,
    start_time: float,
    duration: float,
    amplitude: float = 0.3,
) -> lv.Value:
    """
    Refactored PianoNote.
    This class was already "GOOD"  and is just
    converted to a compositional factory function.
    """

    #
    ### Create ADSR envelope  ###
    #
    envelope: lv.Value = lv.ADSR2(
        time=time,
        note_start=start_time,
        note_duration=duration,
        attack_time=0.02,
        decay_time=0.15,
        sustain_level=0.6,
        release_time=0.3,
    )

    #
    ### Piano harmonics  ###
    #
    pi2: float = 2 * math.pi
    #
    fundamental: lv.Value = lv.Sin(
        value=time, frequency=lv.Constant(frequency * pi2), amplitude=lv.Constant(1.0)
    )
    #
    harmonic2: lv.Value = lv.Sin(
        value=time,
        frequency=lv.Constant(frequency * 2 * pi2),
        amplitude=lv.Constant(0.4),
    )
    #
    harmonic3: lv.Value = lv.Sin(
        value=time,
        frequency=lv.Constant(frequency * 3 * pi2),
        amplitude=lv.Constant(0.2),
    )
    #
    harmonic4: lv.Value = lv.Sin(
        value=time,
        frequency=lv.Constant(frequency * 4 * pi2),
        amplitude=lv.Constant(0.1),
    )

    #
    ### Pre-build the sum of harmonics  ###
    #
    harmonic_sum: lv.Value = lv.Sum([fundamental, harmonic2, harmonic3, harmonic4])

    #
    ### Final = Amplitude * Envelope * Signal ###
    #
    return lv.Product(lv.c(amplitude), envelope, harmonic_sum)


#
def PianoNote2(
    time: lv.Value, frequency: float, start_time: float, duration: float = 2.0
) -> lv.Value:
    """
    Refactored PianoNote2.
    Builds graph from lib_value components.
    The "POOR"  is
    approximated with ExponentialADSR.
    """

    #
    ### Envelope: Approximates the 4-stage original  ###
    #
    amp_env: lv.Value = lv.ExponentialADSR(
        time,
        note_start=start_time,
        note_duration=duration,
        attack_time=0.01,
        decay_time=0.09,
        sustain_level=0.7,
        release_time=0.5,
        attack_curve=1.0,
        decay_curve=1.0,
    )

    #
    ### Harmonics  ###
    #
    relative_time: lv.Value = lv.BasicScaling(time, lv.c(1), lv.c(-start_time))
    pi2: float = 2 * math.pi
    #
    h1: lv.Value = lv.Sin(relative_time, lv.c(frequency * 1 * pi2), lv.c(1.0))
    h2: lv.Value = lv.Sin(relative_time, lv.c(frequency * 2 * pi2), lv.c(0.5))
    h3: lv.Value = lv.Sin(relative_time, lv.c(frequency * 3 * pi2), lv.c(0.25))
    h4: lv.Value = lv.Sin(relative_time, lv.c(frequency * 4 * pi2), lv.c(0.15))
    h5: lv.Value = lv.Sin(relative_time, lv.c(frequency * 5 * pi2), lv.c(0.1))

    #
    signal: lv.Value = lv.Sum(h1, h2, h3, h4, h5)

    #
    ### Final = 0.3 * AmpEnv * Signal ###
    #
    return lv.Product(lv.c(0.3), amp_env, signal)
