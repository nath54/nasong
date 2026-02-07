import nasong.core.value as lv


def TrainableSawtoothSynth(
    time: lv.Value, frequency: lv.Value, start_time: float, duration: float
) -> lv.Value:
    """
    Trainable sawtooth-based synthesizer.

    Trainable parameters:
    - Base amplitude
    - Attack, decay, sustain, release times and levels
    - Harmonics count
    - Filter cutoff/resonance (future)
    """

    # Trainable parameters
    amplitude = lv.ValueTrainableParameter(0.3)
    attack_time = lv.ValueTrainableParameter(0.01)
    decay_time = lv.ValueTrainableParameter(0.1)
    sustain_level = lv.ValueTrainableParameter(0.7)
    release_time = lv.ValueTrainableParameter(0.2)
    num_harmonics = 20  # Fixed for band-limited synthesis

    # Oscillator
    osc = lv.BandLimitedSawtooth(
        time=time, frequency=frequency, amplitude=amplitude, num_harmonics=num_harmonics
    )

    # Envelope
    env = lv.ExponentialADSR(
        time=time,
        note_start=start_time,
        note_duration=duration,
        attack_time=attack_time.value.item(),
        decay_time=decay_time.value.item(),
        sustain_level=sustain_level.value.item(),
        release_time=release_time.value.item(),
        attack_curve=0.5,
        decay_curve=2.0,
        release_curve=2.0,
    )

    return lv.Product(osc, env)


def TrainableSquareSynth(
    time: lv.Value, frequency: lv.Value, start_time: float, duration: float
) -> lv.Value:
    """
    Trainable square wave synthesizer.
    """

    # Trainable parameters
    amplitude = lv.ValueTrainableParameter(0.3)
    attack_time = lv.ValueTrainableParameter(0.01)
    decay_time = lv.ValueTrainableParameter(0.1)
    sustain_level = lv.ValueTrainableParameter(0.7)
    release_time = lv.ValueTrainableParameter(0.2)
    _duty_cycle = lv.ValueTrainableParameter(0.5)
    num_harmonics = 15

    # Oscillator
    osc = lv.BandLimitedSquare(
        time=time, frequency=frequency, amplitude=amplitude, num_harmonics=num_harmonics
    )

    # Envelope
    env = lv.ExponentialADSR(
        time=time,
        note_start=start_time,
        note_duration=duration,
        attack_time=attack_time.value.item(),
        decay_time=decay_time.value.item(),
        sustain_level=sustain_level.value.item(),
        release_time=release_time.value.item(),
    )

    return lv.Product(osc, env)


def TrainableSineSynth(
    time: lv.Value, frequency: lv.Value, start_time: float, duration: float
) -> lv.Value:
    """
    Pure sine wave synthesizer with trainable parameters.
    Good for learning simple tones, bells, pads.
    """

    # Trainable parameters
    amplitude = lv.ValueTrainableParameter(0.5)
    attack_time = lv.ValueTrainableParameter(0.05)
    decay_time = lv.ValueTrainableParameter(0.2)
    sustain_level = lv.ValueTrainableParameter(0.6)
    release_time = lv.ValueTrainableParameter(0.3)

    # Convert frequency to rad/s for Sin
    freq_rads = lv.Product(frequency, lv.Constant(6.283185307179586))  # 2*pi

    # Oscillator
    osc = lv.Sin(value=time, frequency=freq_rads, amplitude=amplitude)

    # Envelope
    env = lv.ExponentialADSR(
        time=time,
        note_start=start_time,
        note_duration=duration,
        attack_time=attack_time.value.item(),
        decay_time=decay_time.value.item(),
        sustain_level=sustain_level.value.item(),
        release_time=release_time.value.item(),
    )

    return lv.Product(osc, env)
