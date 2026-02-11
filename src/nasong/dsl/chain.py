"""
TODO: add full docstring, explaining what the goal of this script is, and explaining for each class and each function what is it, how it works, and how to use it.


Signal Chaining DSL.
Allows syntax like: Osc(freq) >> Filter(cutoff) >> Reverb()
"""

#
### Import Modules. ###
#
from nasong.core.value import Value
from nasong.core.values.mult_itms_ops.value_product import Product
from nasong.core.values.basic.value_constant import Constant


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


class Gain(Processor):
    def __init__(self, amount: float | Value):
        self.amount = amount if isinstance(amount, Value) else Constant(amount)

    def __call__(self, source: Value) -> Value:
        return Product([source, self.amount])


# More effects to be added...
