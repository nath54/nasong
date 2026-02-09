from typing import Callable
from nasong.core.value import Value


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
