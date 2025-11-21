#
### Import Modules. ###
#
import math
import random
#
import lib_value as lv


#
### CATEGORY: BOWED STRINGS ###
#

#
def Violin(
    time: lv.Value,
    frequency: float,
    start_time: float,
    duration: float,
    amplitude: float = 0.15, # Reduced from 0.3
    vibrato_rate: float = 6.0,
    vibrato_depth: float = 0.015,
    sample_rate: int = 44100
) -> lv.Value:

    """
    Violin instrument using Formant-Shaped Additive Synthesis.
    Simulates the rich body resonance of a violin.
    """

    #
    ### 1. Vibrato (LFO on Pitch) ###
    ### Delayed start for realism (starts after 0.1s) ###
    #
    vibrato_lfo: lv.Value = lv.LFO(
        time,
        rate_hz=lv.c(vibrato_rate),
        waveform_class=lv.Sin,
        amplitude=lv.c(vibrato_depth * frequency)
    )
    
    #
    ### Vibrato delay envelope (0 to 1 over 0.5s) ###
    #
    vib_env: lv.Value = lv.ADSR2(
        time, start_time, duration, 0.2, 0.001, 1.0, 0.1
    )
    #
    delayed_vibrato: lv.Value = lv.Product(vibrato_lfo, vib_env)
    
    #
    ### Modulated Frequency ###
    #
    mod_freq: lv.Value = lv.Sum(lv.c(frequency), delayed_vibrato)

    #
    ### 2. Formant Definitions (Approximate Violin Body Resonances) ###
    #
    formants: list[lv.Formant] = [
        lv.Formant(freq=280.0, gain_db=0.0, q=10.0),   # Main Air Resonance
        lv.Formant(freq=450.0, gain_db=-3.0, q=8.0),   # Main Wood Resonance
        lv.Formant(freq=1000.0, gain_db=-6.0, q=5.0),  # Bridge/Body
        lv.Formant(freq=2500.0, gain_db=-12.0, q=3.0), # High sheen
        lv.Formant(freq=4000.0, gain_db=-15.0, q=2.0)  # Air/Bow sizzle
    ]

    #
    ### 3. Generate Harmonics (Sawtooth-like source shaped by formants) ###
    #
    
    harmonics_list: list[lv.Value] = []
    nyquist_limit: float = sample_rate / 2.0
    pi2: float = 2 * math.pi
    num_harmonics: int = 30 # Rich spectrum
    
    for n in range(1, num_harmonics + 1):
        harmonic_freq_static = frequency * n
        if harmonic_freq_static >= nyquist_limit:
            break
            
        # Calculate Formant Gain
        combined_gain: float = 0.0
        for f in formants:
            f_ratio = harmonic_freq_static / f.freq
            denom = math.sqrt( (1 - f_ratio**2)**2 + (f_ratio / f.q)**2 )
            if denom > 0:
                combined_gain += f.gain / denom
        
        # Sawtooth falloff (1/n)
        source_amp = 1.0 / n
        final_amp = source_amp * combined_gain
        
        # Create Oscillator with Modulated Frequency
        # freq_n = mod_freq * n
        freq_n_rad = lv.Product(mod_freq, lv.c(n * pi2))
        
        # Random phase
        delta = lv.c(random.uniform(0, pi2))
        
        harmonics_list.append(
            lv.Sin(
                value=time,
                frequency=freq_n_rad,
                amplitude=lv.c(final_amp),
                delta=delta
            )
        )
        
    # Scale down the sum of harmonics significantly to prevent clipping
    signal: lv.Value = lv.Product(lv.Sum(harmonics_list), lv.c(0.1))

    #
    ### 4. Bow Noise (Scraping sound on attack) ###
    #
    noise: lv.Value = lv.WhiteNoise(seed=int(frequency), scale=0.02) # Reduced noise scale
    # Short burst at start
    noise_env: lv.Value = lv.ADSR2(
        time, start_time, 0.1, 0.01, 0.05, 0.0, 0.01
    )
    bow_noise: lv.Value = lv.Product(noise, noise_env)
    
    # Mix signal and noise
    mixed_signal: lv.Value = lv.Sum(signal, bow_noise)

    #
    ### 5. Amplitude Envelope (Slow attack for bowed feel) ###
    #
    amp_env: lv.Value = lv.ADSR2(
        time,
        note_start=start_time,
        note_duration=duration,
        attack_time=0.1,   # Bow bite
        decay_time=0.1,
        sustain_level=0.9,
        release_time=0.2
    )

    #
    ### 6. Tremolo (Slight amplitude modulation) ###
    #
    tremolo: lv.Value = lv.LFO(
        time, lv.c(vibrato_rate), lv.Sin, amplitude=lv.c(0.1), delta=lv.c(1.0)
    )
    # (1.0 + 0.1*sin) -> varies between 0.9 and 1.1
    tremolo_norm: lv.Value = lv.BasicScaling(tremolo, lv.c(0.5), lv.c(0.5)) # 0.45 to 0.55? No.
    # LFO returns -amp to +amp. 
    # We want 1.0 +/- 0.1.
    # LFO(amp=0.1) -> -0.1 to 0.1.
    # Add 1.0.
    tremolo_final: lv.Value = lv.Sum(tremolo, lv.c(1.0))

    #
    ### Final Output ###
    #
    return lv.Product(
        lv.c(amplitude),
        amp_env,
        tremolo_final,
        mixed_signal
    )


#
def Cello(
    time: lv.Value,
    frequency: float,
    start_time: float,
    duration: float,
    amplitude: float = 0.2, # Reduced from 0.35
    vibrato_rate: float = 5.0, # Slower vibrato for Cello
    vibrato_depth: float = 0.012,
    sample_rate: int = 44100
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
        amplitude=lv.c(vibrato_depth * frequency)
    )
    vib_env: lv.Value = lv.ADSR2(
        time, start_time, duration, 0.25, 0.001, 1.0, 0.1
    )
    delayed_vibrato: lv.Value = lv.Product(vibrato_lfo, vib_env)
    mod_freq: lv.Value = lv.Sum(lv.c(frequency), delayed_vibrato)

    #
    ### 2. Formant Definitions (Cello Body Resonances) ###
    #
    formants: list[lv.Formant] = [
        lv.Formant(freq=100.0, gain_db=0.0, q=8.0),    # Main Air
        lv.Formant(freq=175.0, gain_db=-2.0, q=6.0),   # Main Wood
        lv.Formant(freq=450.0, gain_db=-5.0, q=5.0),   # Body
        lv.Formant(freq=900.0, gain_db=-10.0, q=4.0),  # Upper Body
        lv.Formant(freq=2000.0, gain_db=-20.0, q=2.0)  # Sizzle
    ]

    #
    ### 3. Generate Harmonics ###
    #
    harmonics_list: list[lv.Value] = []
    nyquist_limit: float = sample_rate / 2.0
    pi2: float = 2 * math.pi
    num_harmonics: int = 40 # More harmonics for bass richness
    
    for n in range(1, num_harmonics + 1):
        harmonic_freq_static = frequency * n
        if harmonic_freq_static >= nyquist_limit:
            break
            
        # Calculate Formant Gain
        combined_gain: float = 0.0
        for f in formants:
            f_ratio = harmonic_freq_static / f.freq
            denom = math.sqrt( (1 - f_ratio**2)**2 + (f_ratio / f.q)**2 )
            if denom > 0:
                combined_gain += f.gain / denom
        
        # Sawtooth falloff (1/n)
        source_amp = 1.0 / n
        final_amp = source_amp * combined_gain
        
        # Create Oscillator
        freq_n_rad = lv.Product(mod_freq, lv.c(n * pi2))
        delta = lv.c(random.uniform(0, pi2))
        
        harmonics_list.append(
            lv.Sin(
                value=time,
                frequency=freq_n_rad,
                amplitude=lv.c(final_amp),
                delta=delta
            )
        )
        
    # Scale down harmonics
    signal: lv.Value = lv.Product(lv.Sum(harmonics_list), lv.c(0.1))

    #
    ### 4. Bow Noise ###
    #
    noise: lv.Value = lv.WhiteNoise(seed=int(frequency * 2), scale=0.02) # Reduced noise
    noise_env: lv.Value = lv.ADSR2(
        time, start_time, 0.15, 0.01, 0.05, 0.0, 0.01
    )
    bow_noise: lv.Value = lv.Product(noise, noise_env)
    mixed_signal: lv.Value = lv.Sum(signal, bow_noise)

    #
    ### 5. Amplitude Envelope (Slower attack than violin) ###
    #
    amp_env: lv.Value = lv.ADSR2(
        time,
        note_start=start_time,
        note_duration=duration,
        attack_time=0.15,
        decay_time=0.1,
        sustain_level=0.95,
        release_time=0.3
    )

    #
    ### Final Output ###
    #
    return lv.Product(
        lv.c(amplitude),
        amp_env,
        mixed_signal
    )
