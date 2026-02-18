# Copyright (C) 2026 Nathan Cerisara <https://github.com/nath54/nasong>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""Percussive and rhythmic instrument blueprints for training.

Includes drum-specific topologies (Kick, Snare, Hi-Hat) designed to learn
non-tonal and transient-heavy sounds using a mix of oscillators, noise,
and pitch envelopes.
"""

#
### Import Modules. ###
#
import nasong.core.all_values as lv


def TrainableKick(  # pylint: disable=invalid-name
    time: lv.Value, start_time: float, name_prefix: str = "kick"
) -> lv.Value:
    """Generates a trainable kick drum node.

    Combines a sine wave with a fast pitch sweep (exponential) and a mix of
    white noise and a short high-frequency click for the transient.

    Args:
        time (lv.Value): Master time value.
        start_time (float): Trigger time for the kick beat.
        name_prefix (str): Prefix for trainable params. Defaults to "kick".

    Returns:
        lv.Value: The kick synthesis output node.
    """

    # Trainable parameters
    base_freq = lv.ValueTrainableParameter(60.0, name=f"{name_prefix}_base_freq")  # Hz
    freq_sweep_amount = lv.ValueTrainableParameter(
        40.0, name=f"{name_prefix}_sweep_amt"
    )  # Hz
    decay_rate = lv.ValueTrainableParameter(15.0, name=f"{name_prefix}_decay")
    noise_amount = lv.ValueTrainableParameter(0.1, name=f"{name_prefix}_noise")
    click_amount = lv.ValueTrainableParameter(0.3, name=f"{name_prefix}_click")
    amplitude = lv.ValueTrainableParameter(0.8, name=f"{name_prefix}_amp")

    # Envelope for overall amplitude
    env = lv.ExponentialDecay(
        time=time, start_time=start_time, decay_rate=float(decay_rate.value)
    )

    # Frequency sweep envelope (faster decay)
    pitch_env = lv.ExponentialDecay(time=time, start_time=start_time, decay_rate=30.0)

    # Swept frequency
    freq = lv.Sum(base_freq, lv.Product(freq_sweep_amount, pitch_env))
    freq_rads = lv.Product(freq, lv.Constant(6.283185307179586))

    # Main tone
    tone = lv.Sin(value=time, frequency=freq_rads, amplitude=lv.Constant(1.0))

    # Noise component
    noise = lv.WhiteNoise(seed=42, scale=float(noise_amount.value))

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
        lv.Product(tone, lv.Constant(1.0 - float(noise_amount.value))), noise, click
    )

    return lv.Product(mixed, env, amplitude)


def TrainableSnare(  # pylint: disable=invalid-name
    time: lv.Value, start_time: float, name_prefix: str = "snare"
) -> lv.Value:
    """Creates a trainable snare drum node.

    Mixes a tonal sine component (representing the membrane) with a white
    noise burst (representing the snare wires).

    Args:
        time (lv.Value): Master time.
        start_time (float): Trigger time for the snare beat.
        name_prefix (str): Prefix for identifiers. Defaults to "snare".

    Returns:
        lv.Value: The snare synthesis output node.
    """

    # Trainable parameters
    tone_freq = lv.ValueTrainableParameter(200.0, name=f"{name_prefix}_tone_freq")
    decay_rate = lv.ValueTrainableParameter(20.0, name=f"{name_prefix}_decay")
    noise_amount = lv.ValueTrainableParameter(0.6, name=f"{name_prefix}_noise")
    amplitude = lv.ValueTrainableParameter(0.6, name=f"{name_prefix}_amp")

    # Envelope
    env = lv.ExponentialDecay(
        time=time, start_time=start_time, decay_rate=float(decay_rate.value)
    )

    # Tonal component (membrane modes)
    tone = lv.Sin(
        value=time,
        frequency=lv.Product(tone_freq, lv.Constant(6.283185307179586)),
        amplitude=lv.Constant(1.0 - float(noise_amount.value)),
    )

    # Noise component (snare wires)
    noise = lv.WhiteNoise(seed=7919, scale=float(noise_amount.value))

    # Mix
    mixed = lv.Sum(tone, noise)

    return lv.Product(mixed, env, amplitude)


def TrainableHiHat(  # pylint: disable=invalid-name
    time: lv.Value, start_time: float, is_open: bool = False, name_prefix: str = "hihat"
) -> lv.Value:
    """Creates a trainable hi-hat node.

    Simulates cymbals using white noise and a sum of high-frequency sine waves
    for a 'metallic' texture.

    Args:
        time (lv.Value): Master time.
        start_time (float): Trigger time.
        is_open (bool): Whether to use a longer 'open' decay or short 'closed' one.
        name_prefix (str): Identifier prefix. Defaults to "hihat".

    Returns:
        lv.Value: The hi-hat synthesis output node.
    """

    # Trainable parameters
    base_decay = lv.ValueTrainableParameter(
        30.0 if not is_open else 10.0, name=f"{name_prefix}_decay"
    )
    amplitude = lv.ValueTrainableParameter(0.4, name=f"{name_prefix}_amp")
    brightness = lv.ValueTrainableParameter(1.0, name=f"{name_prefix}_bright")

    # Envelope
    env = lv.ExponentialDecay(
        time=time, start_time=start_time, decay_rate=float(base_decay.value)
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
