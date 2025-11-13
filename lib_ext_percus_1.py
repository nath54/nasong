#
### Import Modules. ###
#
import math
#
import numpy as np
from numpy.typing import NDArray
#
import lib_value as lv


#
def KickDrum(
    time: lv.Value,
    trigger_time: float,
    amplitude: float = 0.6
) -> lv.Value:

    """
    Refactored KickDrum.
    Builds the sound graph from lib_value components.
    """

    #
    ### The original had a 0.5s hard gate. We use ADSR2 for this. ###
    #
    gate_env: lv.Value = lv.ADSR2(
        time, trigger_time, 0.5, 0.001, 0.001, 1.0, 0.001
    )

    #
    ### Amplitude envelope: exp(-relative_time * 8) ###
    #
    amp_env: lv.Value = lv.ExponentialDecay(time, trigger_time, 8.0)

    #
    ### Pitch envelope: 150 * exp(-relative_time * 20) ###
    #
    pitch_env: lv.Value = lv.Product(
        lv.c(150), lv.ExponentialDecay(time, trigger_time, 20.0)
    )

    #
    ### Tone: sin(2 * pi * pitch * relative_time) ###
    #
    relative_time: lv.Value = lv.BasicScaling(time, lv.c(1), lv.c(-trigger_time))
    #
    tone: lv.Value = lv.Sin(
        relative_time,
        frequency=lv.Product(pitch_env, lv.c(2 * math.pi))
    )

    #
    ### Click: 0.3 * exp(-relative_time * 50) ###
    #
    click: lv.Value = lv.Product(
        lv.c(0.3), lv.ExponentialDecay(time, trigger_time, 50.0)
    )

    #
    ### Signal = (Tone + Click) ###
    #
    signal: lv.Value = lv.Sum(tone, click)

    #
    ### Final = Amplitude * Gate * AmpEnv * Signal ###
    #
    return lv.Product(
        lv.c(amplitude),
        gate_env,
        amp_env,
        signal
    )


#
def KickDrum2(time: lv.Value, start_time: float) -> lv.Value:

    """
    Refactored KickDrum2.
    Builds the sound graph from lib_value components.
    """

    #
    ### Gate: 0.5s hard gate. ###
    #
    gate_env: lv.Value = lv.ADSR2(
        time, start_time, 0.5, 0.001, 0.001, 1.0, 0.001
    )

    #
    ### Amplitude envelope: exp(-relative_t * 15) ###
    #
    amp_env: lv.Value = lv.ExponentialDecay(time, start_time, 15.0)

    #
    ### Pitch envelope: 60 + 100 * exp(-relative_t * 50) ###
    #
    pitch_decay: lv.Value = lv.Product(
        lv.c(100), lv.ExponentialDecay(time, start_time, 50.0)
    )
    #
    pitch_env: lv.Value = lv.Sum(lv.c(60), pitch_decay)

    #
    ### Tone: sin(2 * pi * pitch * relative_time) ###
    #
    relative_time: lv.Value = lv.BasicScaling(time, lv.c(1), lv.c(-start_time))
    #
    tone: lv.Value = lv.Sin(
        relative_time,
        frequency=lv.Product(pitch_env, lv.c(2 * math.pi))
    )

    #
    ### Click: 0.5 * exp(-relative_t * 100) * (noise) ###
    #
    click_env: lv.Value = lv.Product(
        lv.c(0.5), lv.ExponentialDecay(time, start_time, 100.0)
    )
    #
    click_noise: lv.Value = lv.WhiteNoise(seed=4567) # Seed from original 
    #
    click: lv.Value = lv.Product(click_env, click_noise)

    #
    ### Signal = (Tone + Click) ###
    #
    signal: lv.Value = lv.Sum(tone, click)

    #
    ### Final = 0.6 * Gate * AmpEnv * Signal ###
    #
    return lv.Product(
        lv.c(0.6),
        gate_env,
        amp_env,
        signal
    )


#
def Snare(
    time: lv.Value,
    trigger_time: float,
    amplitude: float = 0.4
) -> lv.Value:

    """
    Refactored Snare.
    Builds the sound graph from lib_value components.
    Fixes "bad noise".
    """

    #
    ### Gate: 0.3s hard gate. ###
    #
    gate_env: lv.Value = lv.ADSR2(
        time, trigger_time, 0.3, 0.001, 0.001, 1.0, 0.001
    )

    #
    ### Amplitude envelope: exp(-relative_time * 15) ###
    #
    amp_env: lv.Value = lv.ExponentialDecay(time, trigger_time, 15.0)

    #
    ### Tone: 0.4 * sin(2 * pi * 200 * relative_time) ###
    #
    relative_time: lv.Value = lv.BasicScaling(time, lv.c(1), lv.c(-trigger_time))
    #
    tone: lv.Value = lv.Sin(
        relative_time,
        frequency=lv.c(200 * 2 * math.pi),
        amplitude=lv.c(0.4)
    )

    #
    ### Noise: 0.8 * (noise) ###
    #
    noise: lv.Value = lv.Product(
        lv.WhiteNoise(seed=9973), # Seed from original 
        lv.c(0.8)
    )

    #
    ### Signal = (Tone + Noise) ###
    #
    signal: lv.Value = lv.Sum(tone, noise)

    #
    ### Final = Amplitude * Gate * AmpEnv * Signal ###
    #
    return lv.Product(
        lv.c(amplitude),
        gate_env,
        amp_env,
        signal
    )


#
def SnareDrum(time: lv.Value, start_time: float) -> lv.Value:

    """
    Refactored SnareDrum.
    Builds the sound graph from lib_value components.
    Fixes "bad noise".
    """

    #
    ### Gate: 0.2s hard gate. ###
    #
    gate_env: lv.Value = lv.ADSR2(
        time, start_time, 0.2, 0.001, 0.001, 1.0, 0.001
    )

    #
    ### Amplitude envelope: exp(-relative_t * 30) ###
    #
    amp_env: lv.Value = lv.ExponentialDecay(time, start_time, 30.0)

    #
    ### Tone: 0.3 * sin(2 * pi * 200 * relative_time) ###
    #
    relative_time: lv.Value = lv.BasicScaling(time, lv.c(1), lv.c(-start_time))
    #
    tone: lv.Value = lv.Sin(
        relative_time,
        frequency=lv.c(200 * 2 * math.pi),
        amplitude=lv.c(0.3)
    )

    #
    ### Noise: 0.7 * (noise) ###
    #
    noise: lv.Value = lv.Product(
        lv.WhiteNoise(seed=1337), # Seed from original 
        lv.c(0.05)
    )

    #
    ### Signal = (Tone + Noise) ###
    #
    signal: lv.Value = lv.Sum(tone, noise)

    #
    ### Final = 0.4 * Gate * AmpEnv * Signal ###
    #
    return lv.Product(
        lv.c(0.4),
        gate_env,
        amp_env,
        signal
    )


#
def HiHat(time: lv.Value, start_time: float, open: bool = False) -> lv.Value:

    """
    Refactored HiHat.
    Builds the sound graph from lib_value components.
    Fixes "bad noise".
    """

    #
    duration: float = 0.4 if open else 0.08
    decay_rate: float = 8.0 if open else 50.0

    #
    ### Gate: hard gate at duration. ###
    #
    gate_env: lv.Value = lv.ADSR2(
        time, start_time, duration, 0.001, 0.001, 1.0, 0.001
    )

    #
    ### Amplitude envelope: exp(-relative_t * decay_rate) ###
    #
    amp_env: lv.Value = lv.ExponentialDecay(time, start_time, decay_rate)

    #
    ### Tone: Sum of high-frequency sines ###
    #
    relative_time: lv.Value = lv.BasicScaling(time, lv.c(1), lv.c(-start_time))
    pi2: float = 2 * math.pi
    cymbal_freqs: list[float] = [8000.0, 9000.0, 10000.0, 11000.0, 12000.0]
    #
    tone_list: list[lv.Value] = [
        lv.Sin(relative_time, lv.c(f * pi2)) for f in cymbal_freqs
    ]
    #
    tone: lv.Value = lv.Product(
        lv.Sum(tone_list), lv.c(1.0 / len(cymbal_freqs))
    )

    #
    ### Noise: 0.3 * (noise) ###
    #
    noise: lv.Value = lv.Product(
        lv.WhiteNoise(seed=2222), # Seed from original 
        lv.c(0.05)
    )

    #
    ### Signal = (Tone + Noise) ###
    #
    signal: lv.Value = lv.Sum(tone, noise)

    #
    ### Final = 0.15 * Gate * AmpEnv * Signal ###
    #
    return lv.Product(
        lv.c(0.15),
        gate_env,
        amp_env,
        signal
    )


#
def CrashCymbal(time: lv.Value, start_time: float) -> lv.Value:

    """
    Refactored CrashCymbal.
    Builds the sound graph from lib_value components.
    Fixes "bad noise" by using static random phase.
    """

    #
    ### Gate: 2.0s hard gate. ###
    #
    gate_env: lv.Value = lv.ADSR2(
        time, start_time, 2.0, 0.001, 0.001, 1.0, 0.001
    )

    #
    ### Amplitude envelope: exp(-relative_t * 2) ###
    #
    amp_env: lv.Value = lv.ExponentialDecay(time, start_time, 2.0)

    #
    ### Tone: Sum of high-frequency sines with random phase ###
    #
    relative_time: lv.Value = lv.BasicScaling(time, lv.c(1), lv.c(-start_time))
    pi2: float = 2 * math.pi
    #
    cymbal_freqs: list[float] = [
        7000.0, 8500.0, 10000.0, 11500.0, 13000.0, 14500.0
    ]
    #
    ### Pre-calculate static random phases, as in original  ###
    #
    static_phases: NDArray[np.float32] = np.random.uniform(
        low=0.0, high=pi2, size=len(cymbal_freqs)
    ).astype(np.float32)
    #
    tone_list: list[lv.Value] = [
        lv.Sin(relative_time, lv.c(f * pi2), delta=lv.c(float(phase)))
        for f, phase in zip(cymbal_freqs, static_phases)
    ]
    #
    tone: lv.Value = lv.Product(
        lv.Sum(tone_list), lv.c(1.0 / len(cymbal_freqs))
    )

    #
    ### Final = 0.25 * Gate * AmpEnv * Tone ###
    #
    return lv.Product(
        lv.c(0.25),
        gate_env,
        amp_env,
        tone
    )
