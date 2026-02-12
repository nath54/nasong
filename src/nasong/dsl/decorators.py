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


"""DSL decorators for instrument and effect registration.

This module provides decorators that mark functions as specialized components
within the NaSong DSL. These markers are used for discovery, validation,
and potential integration with UI components.
"""

#
### Import Modules. ###
#
from typing import Callable


def instrument(func: Callable) -> Callable:
    """Decorator to register a function as an Instrument.

    Instruments are expected to return a `Value` graph when called. This
    decorator marks the function for discovery by the NaSong ecosystem.

    Args:
        func (Callable): The function to be registered.

    Returns:
        Callable: The decorated function with an `_is_nasong_instrument` flag.
    """
    # Just a marker for now, maybe validation later
    # or registering to a global registry for the TUI to discover.
    func._is_nasong_instrument = True
    return func


def effect(func: Callable) -> Callable:
    """Decorator to register a function as an Effect.

    Effects are expected to take a source `Value` as their first argument and
    return a processed `Value`.

    Args:
        func (Callable): The function to be registered.

    Returns:
        Callable: The decorated function with an `_is_nasong_effect` flag.
    """
    func._is_nasong_effect = True
    return func
