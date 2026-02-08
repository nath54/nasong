import nasong.core.all_values as lv


def TrainableSawtoothSynth(
    time: lv.Value,
    frequency: lv.Value,
    start_time: float,
    duration: float,
    init_amplitude: float = 0.3,
    name_prefix: str = "saw_synth",
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
    amplitude = lv.Constant(init_amplitude)
    attack_time = lv.ValueTrainableParameter(0.01, name=f"{name_prefix}_attack")
    decay_time = lv.ValueTrainableParameter(0.1, name=f"{name_prefix}_decay")
    sustain_level = lv.ValueTrainableParameter(0.7, name=f"{name_prefix}_sustain")
    release_time = lv.ValueTrainableParameter(0.2, name=f"{name_prefix}_release")
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
        attack_time=float(attack_time.value),
        decay_time=float(decay_time.value),
        sustain_level=float(sustain_level.value),
        release_time=float(release_time.value),
        attack_curve=0.5,
        decay_curve=2.0,
        release_curve=2.0,
    )

    return lv.Product(osc, env)


def TrainableSquareSynth(
    time: lv.Value,
    frequency: lv.Value,
    start_time: float,
    duration: float,
    init_amplitude: float = 0.3,
    name_prefix: str = "sq_synth",
) -> lv.Value:
    """
    Trainable square wave synthesizer.
    """

    # Trainable parameters
    amplitude = lv.Constant(init_amplitude)
    attack_time = lv.ValueTrainableParameter(0.01, name=f"{name_prefix}_attack")
    decay_time = lv.ValueTrainableParameter(0.1, name=f"{name_prefix}_decay")
    sustain_level = lv.ValueTrainableParameter(0.7, name=f"{name_prefix}_sustain")
    release_time = lv.ValueTrainableParameter(0.2, name=f"{name_prefix}_release")
    _duty_cycle = lv.ValueTrainableParameter(0.5, name=f"{name_prefix}_duty")
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
        attack_time=float(attack_time.value),
        decay_time=float(decay_time.value),
        sustain_level=float(sustain_level.value),
        release_time=float(release_time.value),
    )

    return lv.Product(osc, env)


def TrainableSineSynth(
    time: lv.Value,
    frequency: lv.Value,
    start_time: float,
    duration: float,
    init_amplitude: float = 0.5,
    name_prefix: str = "sine_synth",
) -> lv.Value:
    """
    Pure sine wave synthesizer with trainable parameters.
    Good for learning simple tones, bells, pads.
    """

    # Trainable parameters
    amplitude = lv.Constant(init_amplitude)
    attack_time = lv.ValueTrainableParameter(0.05, name=f"{name_prefix}_attack")
    decay_time = lv.ValueTrainableParameter(0.2, name=f"{name_prefix}_decay")
    sustain_level = lv.ValueTrainableParameter(0.6, name=f"{name_prefix}_sustain")
    release_time = lv.ValueTrainableParameter(0.3, name=f"{name_prefix}_release")

    # Convert frequency to rad/s for Sin
    freq_rads = lv.Product(frequency, lv.Constant(6.283185307179586))  # 2*pi

    # Oscillator
    osc = lv.Sin(value=time, frequency=freq_rads, amplitude=amplitude)

    # Envelope
    env = lv.ExponentialADSR(
        time=time,
        note_start=start_time,
        note_duration=duration,
        attack_time=float(attack_time.value),
        decay_time=float(decay_time.value),
        sustain_level=float(sustain_level.value),
        release_time=float(release_time.value),
    )

    return lv.Product(osc, env)
