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


"""Instrument registry and lookup for trainable blueprints.

This module maintains a registry of 'trainable' instruments—audio graph nodes
whose parameters can be optimized to match a reference signal. It provides
factory methods to retrieve these blueprints by string identifier.
"""

#
### Import Modules. ###
#
from typing import Callable, Any
from nasong.trainable.instruments import (
    TrainableSawtoothSynth,
    TrainableSquareSynth,
    TrainableSineSynth,
    TrainableKick,
    TrainableSnare,
    TrainableHiHat,
    TrainablePlucked,
    TrainablePiano,
    TrainableBowed,
    TrainablePad,
    TrainableBass,
    TrainableElectricGuitar,
    TrainableUniversalFF,
)

#
### INSTRUMENT REGISTRY ###
#

# Dictionary of all available trainable instruments.
# Maps instrument names to their respective factory functions/classes.
TRAINABLE_INSTRUMENTS: dict[str, Callable[..., Any]] = {
    # Synthesis
    "sawtooth": TrainableSawtoothSynth,
    "square": TrainableSquareSynth,
    "sine": TrainableSineSynth,
    # Percussion
    "kick": TrainableKick,
    "snare": TrainableSnare,
    "hihat_closed": lambda t, f, st, d: TrainableHiHat(t, st, is_open=False),
    "hihat_open": lambda t, f, st, d: TrainableHiHat(t, st, is_open=True),
    # Melodic
    "plucked": TrainablePlucked,
    "piano": TrainablePiano,
    "bowed": TrainableBowed,
    "electric_guitar": TrainableElectricGuitar,
    "universal_ff": TrainableUniversalFF,
    # Atmospheric
    "pad": TrainablePad,
    # Bass
    "bass": TrainableBass,
}


def get_trainable_instrument(instrument_name: str) -> Callable[..., Any]:
    """Retrieves a trainable instrument blueprint by its registry name.

    Args:
        instrument_name (str): The name identifier (e.g., 'sine', 'piano').

    Returns:
        Callable: A function or class that can instantiate the instrument.

    Raises:
        ValueError: If the instrument name is not found in the registry.
    """
    if instrument_name not in TRAINABLE_INSTRUMENTS:
        raise ValueError(
            f"Unknown instrument: {instrument_name}.\
              Available: {list(TRAINABLE_INSTRUMENTS.keys())}"
        )

    return TRAINABLE_INSTRUMENTS[instrument_name]
