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


"""Atmospheric and ambient instrument blueprints for training.

This module provides synthesizers designed for long, evolving sounds, pads,
and textures, with trainable envelopes and detuning parameters.
"""

#
### Import Modules. ###
#
import nasong.core.all_values as lv


def TrainablePad(
    time: lv.Value,
    frequency: lv.Value,
    start_time: float,
    duration: float,
    init_amplitude: float = 0.3,
    name_prefix: str = "pad",
) -> lv.Value:
    """Creates a trainable pad synthesizer node.

    This instrument generates warm, sustained atmospheric sounds using three
    detuned oscillators and an exponential ADSR envelope with slow attack
    and release.

    Args:
        time (lv.Value): The master time value (normalized to seconds).
        frequency (lv.Value): The fundamental frequency of the note.
        start_time (float): Trigger time for the note in seconds.
        duration (float): Length of the note in seconds.
        init_amplitude (float): Starting gain multiplier. Defaults to 0.3.
        name_prefix (str): Prefix for all trainable parameter names.
            Defaults to "pad".

    Returns:
        lv.Value: The output node of the synthesis graph.
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
