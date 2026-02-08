import nasong.core.all_values as lv


def TrainableNamedExamples(
    time: lv.Value, frequency: lv.Value, start_time: float, duration: float
) -> lv.Value:
    """
    Example of an instrument with named parameters for robust tracking.
    """

    # Using named parameters allows the Experiment Manager to save them by name
    # and load them back regardless of order.

    amplitude = lv.ValueTrainableParameter(0.5, name="base_amplitude")

    # ADSR
    attack = lv.ValueTrainableParameter(0.1, name="env_attack")
    decay = lv.ValueTrainableParameter(0.2, name="env_decay")
    sustain = lv.ValueTrainableParameter(0.7, name="env_sustain")
    release = lv.ValueTrainableParameter(0.5, name="env_release")

    # FM Synthesis
    mod_index = lv.ValueTrainableParameter(2.0, name="fm_index")
    mod_ratio = lv.ValueTrainableParameter(1.5, name="fm_ratio")

    # Modulator
    mod_freq = lv.Product(frequency, mod_ratio)
    modulator = lv.Sin(time, lv.Product(mod_freq, lv.Constant(6.2831853)), mod_index)

    # Carrier
    carrier_freq = lv.Product(frequency, lv.Sum(lv.Constant(1.0), modulator))
    carrier = lv.Sin(time, lv.Product(carrier_freq, lv.Constant(6.2831853)), amplitude)

    # Envelope
    env = lv.ExponentialADSR(
        time=time,
        note_start=start_time,
        note_duration=duration,
        attack_time=float(attack.value),
        decay_time=float(decay.value),
        sustain_level=float(sustain.value),
        release_time=float(release.value),
    )

    return lv.Product(carrier, env)
