import nasong.core.all_values as lv


def TrainablePlucked(
    time: lv.Value, frequency: lv.Value, start_time: float, duration: float
) -> lv.Value:
    """
    Trainable plucked string instrument (guitar, harp-like).

    Uses multiple harmonics with individual decay rates.
    """

    # Trainable parameters
    amplitude = lv.ValueTrainableParameter(0.4)
    pluck_decay = lv.ValueTrainableParameter(8.0)
    _brightness = lv.ValueTrainableParameter(0.7)  # Controls harmonic falloff
    attack_time = lv.ValueTrainableParameter(0.001)

    # Very fast attack (pluck)
    env = lv.ExponentialADSR(
        time=time,
        note_start=start_time,
        note_duration=duration,
        attack_time=attack_time.value.item(),
        decay_time=0.1,
        sustain_level=0.3,
        release_time=0.1,
        attack_curve=0.3,
        decay_curve=3.0,
        release_curve=2.0,
    )

    # Additional exponential decay for pluck character
    pluck_env = lv.ExponentialDecay(
        time=time, start_time=start_time, decay_rate=pluck_decay.value.item()
    )

    # Harmonic-rich oscillator
    osc = lv.BandLimitedSawtooth(
        time=time, frequency=frequency, amplitude=amplitude, num_harmonics=25
    )

    return lv.Product(osc, env, pluck_env)


def TrainablePiano(
    time: lv.Value, frequency: lv.Value, start_time: float, duration: float
) -> lv.Value:
    """
    Simplified trainable piano-like instrument.
    """

    # Trainable parameters
    amplitude = lv.ValueTrainableParameter(0.5)
    attack_time = lv.ValueTrainableParameter(0.002)
    decay_time = lv.ValueTrainableParameter(0.3)
    sustain_level = lv.ValueTrainableParameter(0.4)
    release_time = lv.ValueTrainableParameter(0.5)
    brightness = lv.ValueTrainableParameter(0.8)

    # Fast attack, medium decay
    env = lv.ExponentialADSR(
        time=time,
        note_start=start_time,
        note_duration=duration,
        attack_time=attack_time.value.item(),
        decay_time=decay_time.value.item(),
        sustain_level=sustain_level.value.item(),
        release_time=release_time.value.item(),
        attack_curve=0.3,
        decay_curve=2.0,
        release_curve=2.5,
    )

    # Use multiple sine waves for harmonic content
    freq_rads = lv.Product(frequency, lv.Constant(6.283185307179586))

    fundamental = lv.Sin(time, freq_rads, lv.Constant(1.0))
    harmonic2 = lv.Sin(
        time,
        lv.Product(freq_rads, lv.Constant(2.0)),
        lv.Product(brightness, lv.Constant(0.5)),
    )
    harmonic3 = lv.Sin(
        time,
        lv.Product(freq_rads, lv.Constant(3.0)),
        lv.Product(brightness, lv.Constant(0.3)),
    )

    osc = lv.Sum(fundamental, harmonic2, harmonic3)

    return lv.Product(osc, env, amplitude)


def TrainableBowed(
    time: lv.Value, frequency: lv.Value, start_time: float, duration: float
) -> lv.Value:
    """
    Trainable bowed string (violin/cello-like).

    Characteristics: slow attack, sustained tone, rich harmonics.
    """

    # Trainable parameters
    amplitude = lv.ValueTrainableParameter(0.4)
    attack_time = lv.ValueTrainableParameter(0.1)
    decay_time = lv.ValueTrainableParameter(0.2)
    sustain_level = lv.ValueTrainableParameter(0.9)
    release_time = lv.ValueTrainableParameter(0.3)
    vibrato_rate = lv.ValueTrainableParameter(5.0)  # Hz
    vibrato_depth = lv.ValueTrainableParameter(0.02)  # Fraction of frequency

    # Slow attack envelope
    env = lv.ExponentialADSR(
        time=time,
        note_start=start_time,
        note_duration=duration,
        attack_time=attack_time.value.item(),
        decay_time=decay_time.value.item(),
        sustain_level=sustain_level.value.item(),
        release_time=release_time.value.item(),
        attack_curve=2.0,
        decay_curve=1.5,
        release_curve=2.0,
    )

    # Vibrato (LFO modulating frequency)
    vibrato = lv.Sin(
        time, lv.Product(vibrato_rate, lv.Constant(6.283185307179586)), vibrato_depth
    )
    modulated_freq = lv.Product(frequency, lv.Sum(lv.Constant(1.0), vibrato))

    # Rich harmonic content
    osc = lv.BandLimitedSawtooth(
        time=time, frequency=modulated_freq, amplitude=amplitude, num_harmonics=30
    )

    return lv.Product(osc, env)
