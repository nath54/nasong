#
### Import Modules. ###
#
import math

#
import nasong.core.value as lv


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
    """
    Refactored SaxophoneNote.
    Builds graph from lib_value components.
    Fixes "bad noise".
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
