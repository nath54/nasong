#
### Import Modules. ###
#
import math
import random

#
import nasong.core.all_values as lv


#
### CATEGORY: BOWED STRINGS ###
#


#
def Violin(
    time: lv.Value,
    frequency: float,
    start_time: float,
    duration: float,
    amplitude: float = 0.18,  # Slightly increased to compensate for attenuation
    vibrato_rate: float = 30.0,
    vibrato_depth: float = 0.15,
    sample_rate: int = 44100,
) -> lv.Value:
    """
    Violin instrument using Formant-Shaped Additive Synthesis.
    Simulates the rich body resonance of a violin.
    """

    #
    ### 1. Vibrato (LFO on Pitch) ###
    #
    vibrato_lfo: lv.Value = lv.LFO(
        time,
        rate_hz=lv.c(vibrato_rate),
        waveform_class=lv.Sin,
        amplitude=lv.c(vibrato_depth * frequency),
    )

    vib_env: lv.Value = lv.ADSR2(time, start_time, duration, 0.2, 0.001, 1.0, 0.1)
    delayed_vibrato: lv.Value = lv.Product(vibrato_lfo, vib_env)
    mod_freq: lv.Value = lv.Sum(lv.c(frequency), delayed_vibrato)

    #
    ### 2. Formant Definitions - ADJUSTED Q values to reduce odd harmonic amp ###
    #
    formants: list[lv.Formant] = [
        lv.Formant(freq=280.0, gain_db=0.0, q=6.0),  # Reduced Q
        lv.Formant(freq=450.0, gain_db=-3.0, q=5.0),  # Reduced Q
        lv.Formant(freq=1000.0, gain_db=-6.0, q=3.0),  # Reduced Q
        lv.Formant(freq=2500.0, gain_db=-12.0, q=2.0),  # Reduced Q
        lv.Formant(freq=4000.0, gain_db=-15.0, q=1.5),  # Reduced Q
    ]

    #
    ### 3. Generate Harmonics - ATTENUATE ODD HARMONICS ###
    #

    harmonics_list: list[lv.Value] = []
    nyquist_limit: float = sample_rate / 2.0
    pi2: float = 2 * math.pi
    num_harmonics: int = 40

    for n in range(1, num_harmonics + 1):
        harmonic_freq_static = frequency * n
        if harmonic_freq_static >= nyquist_limit:
            break

        # Calculate Formant Gain
        combined_gain: float = 0.0
        for f in formants:
            f_ratio = harmonic_freq_static / f.freq
            denom = math.sqrt((1 - f_ratio**2) ** 2 + (f_ratio / f.q) ** 2)
            if denom > 0:
                combined_gain += f.gain / denom

        # Sawtooth falloff (1/n)
        source_amp = 1.0 / n

        # CORRECTED FIX: ATTENUATE odd harmonics instead
        if n % 2 == 1 and n > 1:  # Don't attenuate fundamental
            source_amp *= 0.5  # 50% reduction for odd harmonics (except fundamental)

        final_amp = source_amp * combined_gain

        # Create Oscillator with Modulated Frequency
        freq_n_rad = lv.Product(mod_freq, lv.c(n * pi2))
        delta = lv.c(random.uniform(0, pi2))

        harmonics_list.append(
            lv.Sin(
                value=time, frequency=freq_n_rad, amplitude=lv.c(final_amp), delta=delta
            )
        )

    # Adjusted scaling
    signal: lv.Value = lv.Product(lv.Sum(harmonics_list), lv.c(0.12))

    #
    ### 4. Bow Noise - MINIMAL ###
    #
    noise: lv.Value = lv.WhiteNoise(seed=int(frequency), scale=0.003)
    noise_env: lv.Value = lv.ADSR2(time, start_time, 0.08, 0.01, 0.03, 0.0, 0.01)
    bow_noise: lv.Value = lv.Product(noise, noise_env)
    mixed_signal: lv.Value = lv.Sum(signal, bow_noise)

    #
    ### 5. Amplitude Envelope ###
    #
    amp_env: lv.Value = lv.ADSR2(
        time,
        note_start=start_time,
        note_duration=duration,
        attack_time=0.1,
        decay_time=0.1,
        sustain_level=0.9,
        release_time=0.2,
    )

    #
    ### 6. Tremolo ###
    #
    tremolo: lv.Value = lv.LFO(
        time, lv.c(vibrato_rate), lv.Sin, amplitude=lv.c(0.08), delta=lv.c(1.0)
    )
    tremolo_final: lv.Value = lv.Sum(tremolo, lv.c(1.0))

    #
    ### Final Output ###
    #
    return lv.Product(lv.c(amplitude), amp_env, tremolo_final, mixed_signal)


#
def Cello(
    time: lv.Value,
    frequency: float,
    start_time: float,
    duration: float,
    amplitude: float = 0.22,  # Increased to compensate for attenuation
    vibrato_rate: float = 0.1,
    vibrato_depth: float = 0.032,
    sample_rate: int = 44100,
) -> lv.Value:
    """
    Cello instrument using Formant-Shaped Additive Synthesis.
    Deeper, warmer resonances.
    """

    #
    ### 1. Vibrato ###
    #
    vibrato_lfo: lv.Value = lv.LFO(
        time,
        rate_hz=lv.c(vibrato_rate),
        waveform_class=lv.Sin,
        amplitude=lv.c(vibrato_depth * frequency),
    )
    vib_env: lv.Value = lv.ADSR2(time, start_time, duration, 0.25, 0.001, 1.0, 0.1)
    delayed_vibrato: lv.Value = lv.Product(vibrato_lfo, vib_env)
    mod_freq: lv.Value = lv.Sum(lv.c(frequency), delayed_vibrato)

    #
    ### 2. Formant Definitions - ADJUSTED Q values ###
    #
    formants: list[lv.Formant] = [
        lv.Formant(freq=100.0, gain_db=0.0, q=5.0),  # Reduced Q
        lv.Formant(freq=175.0, gain_db=-2.0, q=4.0),  # Reduced Q
        lv.Formant(freq=450.0, gain_db=-5.0, q=3.0),  # Reduced Q
        lv.Formant(freq=900.0, gain_db=-10.0, q=2.5),  # Reduced Q
        lv.Formant(freq=2000.0, gain_db=-20.0, q=1.5),  # Reduced Q
    ]

    #
    ### 3. Generate Harmonics - ATTENUATE ODD HARMONICS ###
    #
    harmonics_list: list[lv.Value] = []
    nyquist_limit: float = sample_rate / 2.0
    pi2: float = 2 * math.pi
    num_harmonics: int = 50

    for n in range(1, num_harmonics + 1):
        harmonic_freq_static = frequency * n
        if harmonic_freq_static >= nyquist_limit:
            break

        # Calculate Formant Gain
        combined_gain: float = 0.0
        for f in formants:
            f_ratio = harmonic_freq_static / f.freq
            denom = math.sqrt((1 - f_ratio**2) ** 2 + (f_ratio / f.q) ** 2)
            if denom > 0:
                combined_gain += f.gain / denom

        # Sawtooth falloff (1/n)
        source_amp = 1.0 / n

        # CORRECTED FIX: ATTENUATE odd harmonics (except fundamental)
        if n % 2 == 1 and n > 1:
            source_amp *= 0.55  # 45% reduction

        final_amp = source_amp * combined_gain

        # Create Oscillator
        freq_n_rad = lv.Product(mod_freq, lv.c(n * pi2))
        delta = lv.c(random.uniform(0, pi2))

        harmonics_list.append(
            lv.Sin(
                value=time, frequency=freq_n_rad, amplitude=lv.c(final_amp), delta=delta
            )
        )

    # Scale
    signal: lv.Value = lv.Product(lv.Sum(harmonics_list), lv.c(0.11))

    #
    ### 4. Bow Noise - MINIMAL ###
    #
    noise: lv.Value = lv.WhiteNoise(seed=int(frequency * 2), scale=0.003)
    noise_env: lv.Value = lv.ADSR2(time, start_time, 0.12, 0.01, 0.03, 0.0, 0.01)
    bow_noise: lv.Value = lv.Product(noise, noise_env)
    mixed_signal: lv.Value = lv.Sum(signal, bow_noise)

    #
    ### 5. Amplitude Envelope ###
    #
    amp_env: lv.Value = lv.ADSR2(
        time,
        note_start=start_time,
        note_duration=duration,
        attack_time=0.15,
        decay_time=0.1,
        sustain_level=0.95,
        release_time=0.3,
    )

    #
    ### Final Output ###
    #
    return lv.Product(lv.c(amplitude), amp_env, mixed_signal)
