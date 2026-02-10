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
TODO: add full docstring, explaining what the goal of this script is, and explaining for each class and each function what is it, how it works, and how to use it.
"""

#
### Import Modules. ###
#
from typing import Callable


def instrument(func: Callable) -> Callable:
    """
    Decorator to register a function as an Instrument in the DSL.
    Instruments should return a Value graph.
    """
    # Just a marker for now, maybe validation later
    # or registering to a global registry for the TUI to discover.
    func._is_nasong_instrument = True
    return func


def effect(func: Callable) -> Callable:
    """
    Decorator for Effects.
    Effects take a source Value as first argument.
    """
    func._is_nasong_effect = True
    return func
