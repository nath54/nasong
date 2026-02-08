import nasong.core.all_values as lv


def TrainablePad(
    time: lv.Value,
    frequency: lv.Value,
    start_time: float,
    duration: float,
    init_amplitude: float = 0.3,
    name_prefix: str = "pad",
) -> lv.Value:
    """
    Trainable pad synthesizer (warm, sustained, atmospheric).

    Uses detuned oscillators and long envelopes.
    """

    # Trainable parameters
    amplitude = lv.Constant(init_amplitude)
    attack_time = lv.ValueTrainableParameter(0.5, name=f"{name_prefix}_attack")
    decay_time = lv.ValueTrainableParameter(0.3, name=f"{name_prefix}_decay")
    sustain_level = lv.ValueTrainableParameter(0.8, name=f"{name_prefix}_sustain")
    release_time = lv.ValueTrainableParameter(1.0, name=f"{name_prefix}_release")
    detune_amount = lv.ValueTrainableParameter(
        0.01, name=f"{name_prefix}_detune"
    )  # Slight detuning
    _brightness = lv.ValueTrainableParameter(0.6, name=f"{name_prefix}_bright")

    # Very slow attack/release
    env = lv.ExponentialADSR(
        time=time,
        note_start=start_time,
        note_duration=duration,
        attack_time=float(attack_time.value),
        decay_time=float(decay_time.value),
        sustain_level=float(sustain_level.value),
        release_time=float(release_time.value),
        attack_curve=3.0,
        decay_curve=2.0,
        release_curve=3.0,
    )

    freq_rads = lv.Product(frequency, lv.Constant(6.283185307179586))

    # Three detuned oscillators for richness
    osc1 = lv.Sin(time, freq_rads, lv.Constant(0.33))
    osc2 = lv.Sin(
        time,
        lv.Product(freq_rads, lv.Sum(lv.Constant(1.0), detune_amount)),
        lv.Constant(0.33),
    )
    osc3 = lv.Sin(
        time,
        lv.Product(
            freq_rads,
            lv.Sum(lv.Constant(1.0), lv.Product(detune_amount, lv.Constant(-1.0))),
        ),
        lv.Constant(0.33),
    )

    osc = lv.Sum(osc1, osc2, osc3)

    return lv.Product(osc, env, amplitude)
