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


"""Melodic and polyphonic instrument blueprints for training.

Provides blueprints for plucked strings, piano-like tones, and bowed
instruments, each with specific spectral and temporal characteristics
governed by trainable parameters.
"""

#
### Import Modules. ###
#
import nasong.core.all_values as lv


def TrainablePlucked(
    time: lv.Value,
    frequency: lv.Value,
    start_time: float,
    duration: float,
    init_amplitude: float = 0.4,
    name_prefix: str = "plucked",
) -> lv.Value:
    """Creates a trainable plucked string instrument (e.g., guitar, harp).

    Uses a band-limited sawtooth oscillator modulated by both an ADSR envelope
    and a specific exponential decay for the 'pluck' character.

    Args:
        time (lv.Value): The master time node.
        frequency (lv.Value): Frequency of the note.
        start_time (float): When the string is plucked (seconds).
        duration (float): How long the note is held.
        init_amplitude (float): Initial gain. Defaults to 0.4.
        name_prefix (str): Parameter prefix. Defaults to "plucked".

    Returns:
        lv.Value: The plucked string audio graph node.
    """

    # Trainable parameters
    amplitude = lv.Constant(init_amplitude)
    pluck_decay = lv.ValueTrainableParameter(8.0, name=f"{name_prefix}_decay")
    _brightness = lv.ValueTrainableParameter(
        0.7, name=f"{name_prefix}_bright"
    )  # Controls harmonic falloff
    attack_time = lv.ValueTrainableParameter(0.001, name=f"{name_prefix}_attack")

    # Very fast attack (pluck)
    env = lv.ExponentialADSR(
        time=time,
        note_start=start_time,
        note_duration=duration,
        attack_time=float(attack_time.value),
        decay_time=0.1,
        sustain_level=0.3,
        release_time=0.1,
        attack_curve=0.3,
        decay_curve=3.0,
        release_curve=2.0,
    )

    # Additional exponential decay for pluck character
    pluck_env = lv.ExponentialDecay(
        time=time, start_time=start_time, decay_rate=float(pluck_decay.value)
    )

    # Harmonic-rich oscillator
    osc = lv.BandLimitedSawtooth(
        time=time, frequency=frequency, amplitude=amplitude, num_harmonics=25
    )

    return lv.Product(osc, env, pluck_env)


def TrainablePiano(
    time: lv.Value,
    frequency: lv.Value,
    start_time: float,
    duration: float,
    init_amplitude: float = 0.5,
    name_prefix: str = "piano",
) -> lv.Value:
    """Creates a simplified trainable piano-like instrument.

    Generates tones using a combination of a fundamental sine wave and
    the first two integer harmonics, with an ADSR envelope.

    Args:
        time (lv.Value): Master time.
        frequency (lv.Value): Note frequency.
        start_time (float): When the key is pressed.
        duration (float): Length of the press.
        init_amplitude (float): Base gain. Defaults to 0.5.
        name_prefix (str): Parameter prefix. Defaults to "piano".

    Returns:
        lv.Value: The piano synthesis node.
    """

    # Trainable parameters
    amplitude = lv.Constant(init_amplitude)
    attack_time = lv.ValueTrainableParameter(0.002, name=f"{name_prefix}_attack")
    decay_time = lv.ValueTrainableParameter(0.3, name=f"{name_prefix}_decay")
    sustain_level = lv.ValueTrainableParameter(0.4, name=f"{name_prefix}_sustain")
    release_time = lv.ValueTrainableParameter(0.5, name=f"{name_prefix}_release")
    brightness = lv.ValueTrainableParameter(0.8, name=f"{name_prefix}_bright")

    # Fast attack, medium decay
    env = lv.ExponentialADSR(
        time=time,
        note_start=start_time,
        note_duration=duration,
        attack_time=float(attack_time.value),
        decay_time=float(decay_time.value),
        sustain_level=float(sustain_level.value),
        release_time=float(release_time.value),
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
    time: lv.Value,
    frequency: lv.Value,
    start_time: float,
    duration: float,
    init_amplitude: float = 0.4,
    name_prefix: str = "bowed",
) -> lv.Value:
    """Creates a trainable bowed string (e.g., violin, cello).

    Features a slow attack, high sustain, and frequency modulation to
    simulate natural vibrato.

    Args:
        time (lv.Value): Master time.
        frequency (lv.Value): Fundamental frequency.
        start_time (float): Bowing start time.
        duration (float): Length of the bow stroke.
        init_amplitude (float): Master gain. Defaults to 0.4.
        name_prefix (str): Parameter prefix. Defaults to "bowed".

    Returns:
        lv.Value: The bowed instrument node.
    """

    # Trainable parameters
    amplitude = lv.Constant(init_amplitude)
    attack_time = lv.ValueTrainableParameter(0.1, name=f"{name_prefix}_attack")
    decay_time = lv.ValueTrainableParameter(0.2, name=f"{name_prefix}_decay")
    sustain_level = lv.ValueTrainableParameter(0.9, name=f"{name_prefix}_sustain")
    release_time = lv.ValueTrainableParameter(0.3, name=f"{name_prefix}_release")
    vibrato_rate = lv.ValueTrainableParameter(5.0, name=f"{name_prefix}_vib_rate")  # Hz
    vibrato_depth = lv.ValueTrainableParameter(
        0.02, name=f"{name_prefix}_vib_depth"
    )  # Fraction of frequency

    # Slow attack envelope
    env = lv.ExponentialADSR(
        time=time,
        note_start=start_time,
        note_duration=duration,
        attack_time=float(attack_time.value),
        decay_time=float(decay_time.value),
        sustain_level=float(sustain_level.value),
        release_time=float(release_time.value),
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
