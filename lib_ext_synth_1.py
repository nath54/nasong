#
### Import Modules. ###
#
import math
#
import lib_value as lv


#
def SynthLead(
    time: lv.Value,
    frequency: float,
    start_time: float,
    duration: float = 1.0
) -> lv.Value:

    """
    Refactored SynthLead.
    Builds graph from lib_value components.
    Updated to produce a "Smooth Lead" using detuned oscillators (Unison).
    """

    #
    ### ADSR envelope: Slightly softer attack for a smoother feel ###
    #
    amp_env: lv.Value = lv.ADSR2(
        time,
        note_start=start_time,
        note_duration=duration,
        attack_time=0.05,
        decay_time=0.2,
        sustain_level=0.6,
        release_time=0.3
    )

    #
    ### Vibrato LFO: Subtle 4.0 Hz rate, 0.5% depth (reduced from 2%) ###
    #
    vibrato_lfo: lv.Value = lv.LFO(
        time,
        rate_hz=lv.c(1.0),
        waveform_class=lv.Sin,
        amplitude=lv.c(0.00005 * frequency) # 0.5% frequency deviation
    )
    #
    base_freq: lv.Value = lv.Sum(lv.c(frequency), vibrato_lfo)

    #
    ### Oscillators: 3 Detuned Sawtooths (Unison Effect) ###
    ### This creates a "thick" and "smooth" sound instead of a thin buzz. ###
    #

    # 1. Center Oscillator
    osc_center: lv.Value = lv.BandLimitedSawtooth(
        time,
        frequency=base_freq,
        num_harmonics=30
    )

    # 2. Left Oscillator (Slightly Flat)
    freq_flat: lv.Value = lv.Product(base_freq, lv.c(0.9999))
    osc_flat: lv.Value = lv.BandLimitedSawtooth(
        time,
        frequency=freq_flat,
        num_harmonics=30
    )

    # 3. Right Oscillator (Slightly Sharp)
    freq_sharp: lv.Value = lv.Product(base_freq, lv.c(1.0001))
    osc_sharp: lv.Value = lv.BandLimitedSawtooth(
        time,
        frequency=freq_sharp,
        num_harmonics=30
    )

    #
    ### Mix the oscillators: Center is dominant, sides add "smoothness" ###
    #
    mixed_signal: lv.Value = lv.Sum(
        lv.Product(osc_center, lv.c(0.5)),
        lv.Product(osc_flat, lv.c(0.25)),
        lv.Product(osc_sharp, lv.c(0.25))
    )

    #
    ### Final = 0.4 * AmpEnv * MixedSignal ###
    #
    return lv.Product(
        lv.c(0.4),
        amp_env,
        mixed_signal
    )


#
def SynthBass(
    time: lv.Value,
    frequency: float,
    start_time: float,
    duration: float = 0.5
) -> lv.Value:

    """
    Refactored SynthBass.
    Builds graph from lib_value components.
    Fixes "VERY POOR" aliasing using `BandLimitedSquare`.
    """

    #
    ### Envelope: 0.01s attack, then exp(-t * 5) decay ###
    #
    amp_env: lv.Value = lv.ExponentialADSR(
        time,
        note_start=start_time,
        note_duration=duration,
        attack_time=0.01,
        decay_time=duration - 0.01,
        sustain_level=0.0,
        release_time=0.01,
        attack_curve=1.0,     # Linear attack
        decay_curve=2.5       # Exponential decay (approx. exp(-t*5))
    )

    #
    ### Oscillator: Replaced "naive" square with anti-aliased version. ###
    #
    osc_square: lv.Value = lv.BandLimitedSquare(
        time,
        frequency=lv.c(frequency),
        amplitude=lv.c(0.6),
        num_harmonics=20
    )

    #
    ### Sub-oscillator: Sine wave one octave down ###
    #
    relative_time: lv.Value = lv.BasicScaling(time, lv.c(1), lv.c(-start_time))
    #
    osc_sub: lv.Value = lv.Sin(
        relative_time,
        frequency=lv.c(frequency / 2.0 * 2 * math.pi),
        amplitude=lv.c(0.4)
    )

    #
    ### Signal = (Square * 0.6 + Sub * 0.4) ###
    #
    signal: lv.Value = lv.Sum(osc_square, osc_sub)

    #
    ### Final = 0.35 * AmpEnv * Signal ###
    #
    return lv.Product(
        lv.c(0.35),
        amp_env,
        signal
    )


#
def SynthPad(
    time: lv.Value,
    frequency: float,
    start_time: float,
    duration: float = 4.0
) -> lv.Value:

    """
    Refactored SynthPad.
    This class was already "EXCELLENT".
    This is just a 1:1 compositional refactor.
    """

    #
    ### Slow ASR envelope ###
    #
    amp_env: lv.Value = lv.ADSR2(
        time,
        note_start=start_time,
        note_duration=duration,
        attack_time=0.5,
        decay_time=0.001,
        sustain_level=1.0,
        release_time=1.0
    )

    #
    ### Three detuned oscillators ###
    #
    relative_time: lv.Value = lv.BasicScaling(time, lv.c(1), lv.c(-start_time))
    pi2: float = 2 * math.pi
    #
    osc1: lv.Value = lv.Sin(
        relative_time, frequency=lv.c(frequency * pi2)
    )
    #
    osc2: lv.Value = lv.Sin(
        relative_time, frequency=lv.c(frequency * 1.003 * pi2)
    )
    #
    osc3: lv.Value = lv.Sin(
        relative_time, frequency=lv.c(frequency * 0.997 * pi2)
    )

    #
    ### Signal = Average of the three oscillators ###
    #
    signal: lv.Value = lv.Product(
        lv.Sum(osc1, osc2, osc3),
        lv.c(1.0 / 3.0)
    )

    #
    ### Final = 0.2 * AmpEnv * Signal ###
    #
    return lv.Product(
        lv.c(0.2),
        amp_env,
        signal
    )
