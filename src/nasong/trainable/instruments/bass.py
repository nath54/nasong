import nasong.core.all_values as lv


def TrainableBass(
    time: lv.Value, frequency: lv.Value, start_time: float, duration: float
) -> lv.Value:
    """
    Trainable bass synthesizer.

    Rich low-end with controllable harmonics.
    """

    # Trainable parameters
    amplitude = lv.ValueTrainableParameter(0.6)
    attack_time = lv.ValueTrainableParameter(0.005)
    decay_time = lv.ValueTrainableParameter(0.2)
    sustain_level = lv.ValueTrainableParameter(0.6)
    release_time = lv.ValueTrainableParameter(0.1)
    sub_amount = lv.ValueTrainableParameter(0.3)  # Sub-octave amount
    distortion_amount = lv.ValueTrainableParameter(1.5)

    # Punchy envelope
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

    freq_rads = lv.Product(frequency, lv.Constant(6.283185307179586))

    # Main oscillator (sawtooth for richness)
    main_osc = lv.BandLimitedSawtooth(
        time=time, frequency=frequency, amplitude=lv.Constant(0.7), num_harmonics=15
    )

    # Sub-octave sine wave
    sub_osc = lv.Sin(
        time,
        lv.Product(freq_rads, lv.Constant(0.5)),  # One octave down
        sub_amount,
    )

    # Mix oscillators
    mixed = lv.Sum(main_osc, sub_osc)

    # Optional soft distortion for warmth
    distorted = lv.Distortion(value=mixed, gain=distortion_amount)

    return lv.Product(distorted, env, amplitude)
