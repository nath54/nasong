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
Input argument handling utilities for multi-input Value operations.

This module provides the `input_args_to_values` function, which standardizes
how flexible arguments (*args) are converted into a list of `Value` objects.
"""

#
### Import Modules. ###
#
from typing import Any

#
from nasong.core.value import Value
from nasong.core.values.basic.value_constant import Constant


#
def input_args_to_values(values: tuple[Any, ...]) -> list[Value]:
    """Standardizes flexible *args inputs into a list of Value objects.

    This utility allows multi-input classes (like Sum, Product, Min, Max) to
    accept either multiple positional arguments or a single collection of items.
    It also automatically wraps non-Value items (like floats or ints) into
    `Constant` objects.

    Example:
        >>> from nasong.core.values.basic.value_constant import Constant
        >>> v1, v2 = Constant(0.1), Constant(0.2)
        >>> # Both calls below return the same list of Value objects:
        >>> input_args_to_values((v1, v2, 0.5))
        >>> input_args_to_values(([v1, v2, 0.5],))

    Args:
        values: The arguments passed to the class constructor (usually via *args).

    Returns:
        list[Value]: A list of clean Value objects.
    """
    if len(values) == 0:
        return [Constant(value=0)]

    # 1. Determine the raw list of items
    # If the first argument is a collection (and not a Value), assume it's the full list
    if isinstance(values[0], (list, tuple)) and not isinstance(values[0], Value):
        raw_items = values[0]
    else:
        raw_items = values

    # 2. Wrap each item if it's not a Value
    final_values: list[Value] = []
    for item in raw_items:
        if isinstance(item, Value):
            final_values.append(item)
        else:
            # Handle float, int, etc.
            final_values.append(Constant(item))

    return final_values
