"""
Signal Chaining DSL.
Allows syntax like: Osc(freq) >> Filter(cutoff) >> Reverb()
"""

from nasong.core.value import Value
from nasong.core.values.single_itms_ops.value_basic_scaling import (
    BasicScaling,
)  # as Amp?
from nasong.core.values.mult_itms_ops.value_product import Product
from typing import Union, List


class Chainable:
    """
    Mixin or Wrapper to allow >> operator.
    """

    def __init__(self, value: Value):
        self.value = value

    def __rshift__(self, other):
        """
        self >> other
        If other is a function/class taking a source, apply it.
        If other is a Value, maybe mix?
        Usually in audio DSLs, >> means "feed into".

        Case 1: Osc >> Filter
        Filter must be instantiated? Or Filter class?
        If Filter is a class, we instantiate it with self as input.
        If Filter is an instance, can we re-route input? (Hard in static graph)

        Better approach:
        Osc(freq) is a Value.
        Filter(source, cutoff) is a Value.

        We want: Osc(freq) >> Filter(cutoff)
        Here Filter(cutoff) must return a "Partial" or "Processor" that waits for source.

        So:
        class Filter(Processor):
            def __init__(self, cutoff):
                self.cutoff = cutoff
            def __call__(self, source):
                return LowPass(source, self.cutoff)

        Then: Osc >> Filter
        Osc must be Chainable.
        Osc >> Processor -> Processor(Osc)
        """
        if isinstance(other, Processor):
            result_value = other(self.value)
            return Chainable(result_value)

        # If other is just a Value (e.g. gain scaling?)
        # Osc >> 0.5 -> Osc * 0.5
        if isinstance(other, (int, float)):
            # Basic amp
            return Chainable(Product([self.value, Constant(other)]))

        raise TypeError(f"Cannot chain {type(self)} into {type(other)}")

    def val(self) -> Value:
        return self.value


class Processor:
    """
    Base class for effects waiting for an input source.
    """

    def __call__(self, source: Value) -> Value:
        raise NotImplementedError


# We need to wrap NaSong core `Value`s to be chainable?
# Or we can monkey-patch `Value.__rshift__`?
# Monkey-patching is risky but clean for DSL.
# Let's try to define a wrapper `Signal` that inherits from Value?
# Or just use `Chainable` wrapper in the DSL context.

from nasong.core.values.basic.value_constant import Constant


class Gain(Processor):
    def __init__(self, amount: Union[float, Value]):
        self.amount = amount if isinstance(amount, Value) else Constant(amount)

    def __call__(self, source: Value) -> Value:
        return Product([source, self.amount])


# More effects to be added...
