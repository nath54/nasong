import nasong.core.value as lv


def TrainableKick(time: lv.Value, start_time: float) -> lv.Value:
    """
    Trainable kick drum.

    Trainable parameters:
    - Base frequency and pitch sweep
    - Decay rate
    - Noise amount
    - Click amount
    """

    # Trainable parameters
    base_freq = lv.ValueTrainableParameter(60.0)  # Hz
    freq_sweep_amount = lv.ValueTrainableParameter(40.0)  # Hz
    decay_rate = lv.ValueTrainableParameter(15.0)
    noise_amount = lv.ValueTrainableParameter(0.1)
    click_amount = lv.ValueTrainableParameter(0.3)
    amplitude = lv.ValueTrainableParameter(0.8)

    # Envelope for overall amplitude
    env = lv.ExponentialDecay(
        time=time, start_time=start_time, decay_rate=decay_rate.value.item()
    )

    # Frequency sweep envelope (faster decay)
    pitch_env = lv.ExponentialDecay(time=time, start_time=start_time, decay_rate=30.0)

    # Swept frequency
    freq = lv.Sum(base_freq, lv.Product(freq_sweep_amount, pitch_env))
    freq_rads = lv.Product(freq, lv.Constant(6.283185307179586))

    # Main tone
    tone = lv.Sin(value=time, frequency=freq_rads, amplitude=lv.Constant(1.0))

    # Noise component
    noise = lv.WhiteNoise(seed=42, scale=noise_amount.value.item())

    # Click component (very short high-freq burst)
    click_env = lv.ExponentialDecay(time=time, start_time=start_time, decay_rate=100.0)
    click = lv.Product(
        lv.Sin(
            value=time,
            frequency=lv.Constant(1000.0 * 6.283185307179586),
            amplitude=click_amount,
        ),
        click_env,
    )

    # Mix components
    mixed = lv.Sum(
        lv.Product(tone, lv.Constant(1.0 - noise_amount.value.item())), noise, click
    )

    return lv.Product(mixed, env, amplitude)


def TrainableSnare(time: lv.Value, start_time: float) -> lv.Value:
    """
    Trainable snare drum.
    """

    # Trainable parameters
    tone_freq = lv.ValueTrainableParameter(200.0)
    decay_rate = lv.ValueTrainableParameter(20.0)
    noise_amount = lv.ValueTrainableParameter(0.6)
    amplitude = lv.ValueTrainableParameter(0.6)

    # Envelope
    env = lv.ExponentialDecay(
        time=time, start_time=start_time, decay_rate=decay_rate.value.item()
    )

    # Tonal component (membrane modes)
    tone = lv.Sin(
        value=time,
        frequency=lv.Product(tone_freq, lv.Constant(6.283185307179586)),
        amplitude=lv.Constant(1.0 - noise_amount.value.item()),
    )

    # Noise component (snare wires)
    noise = lv.WhiteNoise(seed=7919, scale=noise_amount.value.item())

    # Mix
    mixed = lv.Sum(tone, noise)

    return lv.Product(mixed, env, amplitude)


def TrainableHiHat(
    time: lv.Value, start_time: float, is_open: bool = False
) -> lv.Value:
    """
    Trainable hi-hat (closed or open).
    """

    # Trainable parameters
    base_decay = lv.ValueTrainableParameter(30.0 if not is_open else 10.0)
    amplitude = lv.ValueTrainableParameter(0.4)
    brightness = lv.ValueTrainableParameter(1.0)

    # Envelope
    env = lv.ExponentialDecay(
        time=time, start_time=start_time, decay_rate=base_decay.value.item()
    )

    # Hi-hat is mostly noise with some metallic high frequencies
    noise = lv.WhiteNoise(seed=8191, scale=1.0)

    # Add some high-frequency tones for metallic character
    metallic = lv.Sum(
        lv.Sin(time, lv.Constant(8000.0 * 6.283185307179586), brightness),
        lv.Sin(
            time,
            lv.Constant(10000.0 * 6.283185307179586),
            lv.Product(brightness, lv.Constant(0.7)),
        ),
        lv.Sin(
            time,
            lv.Constant(12000.0 * 6.283185307179586),
            lv.Product(brightness, lv.Constant(0.5)),
        ),
    )

    mixed = lv.Sum(
        lv.Product(noise, lv.Constant(0.7)), lv.Product(metallic, lv.Constant(0.3))
    )

    return lv.Product(mixed, env, amplitude)
