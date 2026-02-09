#
### Import Modules. ###
#
from typing import Any

#
from nasong.core.value import Value
from nasong.core.values.basic.value_constant import Constant


#
def input_args_to_values(values: tuple[Any, ...]) -> list[Value]:
    """
    A utility function to handle flexible *args inputs for multi-input classes
    (like Sum, Min, Max, Product).

    This allows users to pass either `Sum(v1, v2, 0.5)` or `Sum([v1, v2, 0.5])`.

    Args:
        values: The arguments passed to the class constructor (usually *args).

    Returns:
        A clean list of Value objects.
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
