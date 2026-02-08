#
### Import Modules. ###
#
from typing import cast

#
from nasong.core.value import Value
from nasong.core.values.basic.value_constant import Constant


#
def input_args_to_values(values: tuple[Value | list[Value], ...]) -> list[Value]:
    """
    A utility function to handle flexible *args inputs for multi-input classes
    (like Sum, Min, Max, Product).

    This allows users to pass either `Sum(v1, v2, v3)` or `Sum([v1, v2, v3])`.

    Args:
        values: The arguments passed to the class constructor.

    Returns:
        A clean iterable of Value objects.
    """

    #
    if len(values) == 0:
        #
        return [Constant(value=0)]

    #
    if isinstance(values[0], Value):
        #
        return cast(list[Value], list(values))

    #
    return values[0]
