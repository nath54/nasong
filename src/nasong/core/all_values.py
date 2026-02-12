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


"""
Consolidated Value operations registry.

This module imports and exports all `Value` subclasses and utility functions
from the `nasong.core.values` package. It provides a convenient single point of
access for defining complex value-based sounds and modulations.

Example:
    >>> from nasong.core import all_values as v
    >>> song_value = v.Sin(v.Identity(), frequency=v.Constant(440.0))
"""

#
### consolidated values module. ###
#

# From core.value
from nasong.core.value import Value, ValueTrainableParameter, ParameterContext

# From basic/
from nasong.core.values.basic.value_constant import Constant, c
from nasong.core.values.basic.value_identity import Identity
from nasong.core.values.basic.value_random_choice import RandomChoice
from nasong.core.values.basic.value_random_float import RandomFloat
from nasong.core.values.basic.value_random_int import RandomInt
from nasong.core.values.basic.value_white_noise import WhiteNoise

# From complex/
from nasong.core.values.complex.value_adsr2 import ADSR2
from nasong.core.values.complex.value_band_limited_sawtooth import BandLimitedSawtooth
from nasong.core.values.complex.value_band_limited_square import BandLimitedSquare
from nasong.core.values.complex.value_cos import Cos
from nasong.core.values.complex.value_distortion import Distortion
from nasong.core.values.complex.value_exponential_adsr import ExponentialADSR
from nasong.core.values.complex.value_exponential_decay import ExponentialDecay
from nasong.core.values.complex.value_log import Log
from nasong.core.values.complex.value_pow import Pow
from nasong.core.values.complex.value_sawtooth import Sawtooth
from nasong.core.values.complex.value_sin import Sin
from nasong.core.values.complex.value_square import Square
from nasong.core.values.complex.value_triangle import Triangle

# From mult_itms_ops/
from nasong.core.values.mult_itms_ops.value_max import Max
from nasong.core.values.mult_itms_ops.value_min import Min
from nasong.core.values.mult_itms_ops.value_pondered_sum import PonderedSum
from nasong.core.values.mult_itms_ops.value_product import Product
from nasong.core.values.mult_itms_ops.value_sum import Sum

# From single_itms_ops/
from nasong.core.values.single_itms_ops.value_abs import Abs
from nasong.core.values.single_itms_ops.value_basic_scaling import BasicScaling
from nasong.core.values.single_itms_ops.value_clamp import Clamp
from nasong.core.values.single_itms_ops.value_high_pass import HighPass
from nasong.core.values.single_itms_ops.value_low_pass import LowPass
from nasong.core.values.single_itms_ops.value_mask_treshold import MaskTreshold
from nasong.core.values.single_itms_ops.value_modulo import Modulo
from nasong.core.values.single_itms_ops.value_polynom import Polynom
from nasong.core.values.single_itms_ops.value_sequencer import Sequencer
from nasong.core.values.single_itms_ops.value_time_interval import TimeInterval

# Standalone values/
from nasong.core.values.formant import Formant, generate_formant_harmonics
from nasong.core.values.utils import generate_harmonics, LFO
from nasong.core.values.input_args import input_args_to_values
from nasong.core.values.music_theory_and_composition import (
    midi_note_to_freq,
    get_chord_frequencies,
    SimpleMelody,
)

# Define __all__ for clean star imports (optional but good practice)
__all__ = [
    "Value",
    "ValueTrainableParameter",
    "ParameterContext",
    "Constant",
    "c",
    "Identity",
    "RandomChoice",
    "RandomFloat",
    "RandomInt",
    "WhiteNoise",
    "ADSR2",
    "BandLimitedSawtooth",
    "BandLimitedSquare",
    "Cos",
    "Distortion",
    "ExponentialADSR",
    "ExponentialDecay",
    "Log",
    "Pow",
    "Sawtooth",
    "Sin",
    "Square",
    "Triangle",
    "Max",
    "Min",
    "PonderedSum",
    "Product",
    "Sum",
    "Abs",
    "BasicScaling",
    "Clamp",
    "HighPass",
    "LowPass",
    "MaskTreshold",
    "Modulo",
    "Polynom",
    "Sequencer",
    "TimeInterval",
    "Formant",
    "generate_formant_harmonics",
    "generate_harmonics",
    "LFO",
    "input_args_to_values",
    "midi_note_to_freq",
    "get_chord_frequencies",
    "SimpleMelody",
]
