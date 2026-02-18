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


"""Bass instrument blueprints for training.

This module provides bass synthesizers with emphasized low-end and
controllable grit, suitable for learning rhythmic foundation sounds.
"""

#
### Import Modules. ###
#
import nasong.core.all_values as lv


def TrainableBass(  # pylint: disable=invalid-name
    time: lv.Value,
    frequency: lv.Value,
    start_time: float,
    duration: float,
    init_amplitude: float = 0.6,
    name_prefix: str = "bass",
) -> lv.Value:
    """Creates a trainable bass synthesizer node.

    Combines a band-limited sawtooth oscillator with a sub-octave sine wave
    and an optional distortion stage for harmonic saturation.

    Args:
        time (lv.Value): The master time value.
        frequency (lv.Value): The fundamental frequency.
        start_time (float): Trigger time for the note in seconds.
        duration (float): Length of the note in seconds.
        init_amplitude (float): Initial amplitude multiplier. Defaults to 0.6.
        name_prefix (str): Prefix for parameter identifiers. Defaults to "bass".

    Returns:
        lv.Value: The output node of the bass synthesis graph.
    """

    # Trainable parameters
    amplitude = lv.Constant(init_amplitude)
    attack_time = lv.ValueTrainableParameter(0.005, name=f"{name_prefix}_attack")
    decay_time = lv.ValueTrainableParameter(0.2, name=f"{name_prefix}_decay")
    sustain_level = lv.ValueTrainableParameter(0.6, name=f"{name_prefix}_sustain")
    release_time = lv.ValueTrainableParameter(0.1, name=f"{name_prefix}_release")
    sub_amount = lv.ValueTrainableParameter(
        0.3, name=f"{name_prefix}_sub"
    )  # Sub-octave amount
    distortion_amount = lv.ValueTrainableParameter(1.5, name=f"{name_prefix}_dist")

    # Punchy envelope
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
