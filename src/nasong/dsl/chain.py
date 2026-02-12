"""Signal Chaining Domain Specific Language (DSL).

This module provides a fluent interface for constructing audio processing graphs
using the `>>` operator. It allows for a concise "functional" style of defining
synthesizers and effect chains.

Example:
    >>> from nasong.dsl.chain import Chainable, Gain
    >>> from nasong.core.all_values import Sin, Constant
    >>> # Create a sine wave and feed it into a gain stage
    >>> result = Chainable(Sin(Constant(440))) >> Gain(0.5)
    >>> audio_value = result.val()
"""

#
### Import Modules. ###
#
from nasong.core.value import Value
from nasong.core.values.mult_itms_ops.value_product import Product
from nasong.core.values.basic.value_constant import Constant


class Chainable:
    """Wrapper that enables signal chaining via the `>>` operator.

    A `Chainable` object encapsulates a `Value` and provides an implementation
    of `__rshift__` to feed that value into `Processor` objects or apply
    scalar gain.
    """

    def __init__(self, value: Value):
        """Initializes the Chainable wrapper.

        Args:
            value (Value): The audio value to be wrapped.
        """
        self.value = value

    def __rshift__(self, other):
        """Implements the `>>` operator for "feeding" signals.

        If `other` is a `Processor`, it is called with the current value as
        the source. If `other` is a numeric value, it is treated as a gain
        scaling factor.

        Args:
            other (Processor | int | float): The target processor or gain value.

        Returns:
            Chainable: A new Chainable wrapper containing the resulting value.

        Raises:
            TypeError: If `other` is not a Processor or a numeric type.
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
        """Unwraps and returns the underlying `Value` object.

        Returns:
            Value: The accumulated audio value tree.
        """
        return self.value


class Processor:
    """Base class for signal processors in the DSL.

    Processors are "partial" functions that wait for a source `Value` to be
    passed via the `__call__` method (typically triggered by `>>`).
    """

    def __call__(self, source: Value) -> Value:
        """Processes the input source and returns a new Value.

        Args:
            source (Value): The input audio signal.

        Returns:
            Value: The processed audio signal.
        """
        raise NotImplementedError


class Gain(Processor):
    """A simple processor that scales the amplitude of a signal."""

    def __init__(self, amount: float | Value):
        self.amount = amount if isinstance(amount, Value) else Constant(amount)

    def __call__(self, source: Value) -> Value:
        return Product([source, self.amount])


# More effects to be added...
