#
### Import Modules. ###
#
import nasong.core.value as lv
#

#
### TRAINABLE INSTRUMENT BLUEPRINTS ###
#
# All hyper-parameters are ValueTrainableParameter for gradient-based learning.
# Notes and durations are passed as parameters to the blueprint functions.
#


#
### 1. BASIC SYNTHESIS INSTRUMENTS ###
#


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


#
### 2. PERCUSSION INSTRUMENTS ###
#


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


#
### 3. MELODIC INSTRUMENTS ###
#


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


#
### 4. PAD/ATMOSPHERIC INSTRUMENTS ###
#


def TrainablePad(
    time: lv.Value, frequency: lv.Value, start_time: float, duration: float
) -> lv.Value:
    """
    Trainable pad synthesizer (warm, sustained, atmospheric).

    Uses detuned oscillators and long envelopes.
    """

    # Trainable parameters
    amplitude = lv.ValueTrainableParameter(0.3)
    attack_time = lv.ValueTrainableParameter(0.5)
    decay_time = lv.ValueTrainableParameter(0.3)
    sustain_level = lv.ValueTrainableParameter(0.8)
    release_time = lv.ValueTrainableParameter(1.0)
    detune_amount = lv.ValueTrainableParameter(0.01)  # Slight detuning
    _brightness = lv.ValueTrainableParameter(0.6)

    # Very slow attack/release
    env = lv.ExponentialADSR(
        time=time,
        note_start=start_time,
        note_duration=duration,
        attack_time=attack_time.value.item(),
        decay_time=decay_time.value.item(),
        sustain_level=sustain_level.value.item(),
        release_time=release_time.value.item(),
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


#
### 5. BASS INSTRUMENTS ###
#


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


#
### INSTRUMENT REGISTRY ###
#

# Dictionary of all available trainable instruments
TRAINABLE_INSTRUMENTS = {
    # Synthesis
    "sawtooth": TrainableSawtoothSynth,
    "square": TrainableSquareSynth,
    "sine": TrainableSineSynth,
    # Percussion
    "kick": TrainableKick,  # Note: kick doesn't take frequency
    "snare": TrainableSnare,  # Note: snare doesn't take frequency
    "hihat_closed": lambda t, f, st, d: TrainableHiHat(t, st, is_open=False),
    "hihat_open": lambda t, f, st, d: TrainableHiHat(t, st, is_open=True),
    # Melodic
    "plucked": TrainablePlucked,
    "piano": TrainablePiano,
    "bowed": TrainableBowed,
    # Atmospheric
    "pad": TrainablePad,
    # Bass
    "bass": TrainableBass,
}


def get_trainable_instrument(instrument_name: str):
    """
    Get a trainable instrument blueprint by name.

    Args:
        instrument_name: Name of instrument from TRAINABLE_INSTRUMENTS

    Returns:
        Instrument blueprint function
    """
    if instrument_name not in TRAINABLE_INSTRUMENTS:
        raise ValueError(
            f"Unknown instrument: {instrument_name}. Available: {list(TRAINABLE_INSTRUMENTS.keys())}"
        )

    return TRAINABLE_INSTRUMENTS[instrument_name]
