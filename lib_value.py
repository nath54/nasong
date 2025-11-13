#
### Import Modules. ###
#
from typing import cast
#
import random
import math
#
import numpy as np
from numpy.typing import NDArray


#
### ABSTRACT CLASS. ###
#

#
class Value:

    """
    Abstract base class for a time-varying value.

    This class defines the interface for all 'Value' objects, which are used
    to generate signals, envelopes, modulations, etc., on a per-sample basis.
    """

    #
    def __init__(self) -> None:

        """Initializes the base Value object."""

        #
        pass

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        """
        Get the value at a single sample index.

        This is the non-vectorized, sample-by-sample method.
        It's often slower and used as a fallback.

        Args:
            index: The sample index (integer).

        Returns:
            The calculated value (float) at that index.
        """

        #
        return 0

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        """
        Get the values for an array of sample indexes (vectorized).

        This is the performance-critical method used for rendering audio blocks.

        For the implementation of the base Value class:

            - The base implementation is a slow, non-optimized placeholder
            that iterates and calls __getitem__.

            - Subclasses should override this
            with a fast, vectorized NumPy implementation.

        Args:
            indexes_buffer: A NumPy array of sample indexes (as floats).

        Returns:
            A NumPy array of calculated values (float32), matching the
            shape of indexes_buffer.
        """

        #
        ### If we arrive here, it is because there are not implemented getitem_np method, so we are using this non optimized placeholder. ###
        #
        default: NDArray[np.float32] = np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        for idx, i in enumerate(indexes_buffer):
            #
            default[idx] = self.__getitem__(index=int(i), sample_rate=sample_rate)

        #
        return default


#
### BASIC INSTANCES CLASSES. ###
#

#
class Constant(Value):

    """A Value that returns the same constant number for all indexes."""

    #
    def __init__(self, value: float | int) -> None:

        #
        super().__init__()

        #
        self.value: float | int = value

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        return self.value

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        return np.full_like(indexes_buffer, fill_value=self.value, dtype=np.float32)


#
class Identity(Value):

    """A Value that returns the sample index itself as the value."""

    #
    def __init__(self) -> None:

        #
        super().__init__()

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        return float(index)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        return indexes_buffer


#
class RandomInt(Value):

    """
    A Value that returns a random integer within a specified range
    for each sample.
    """

    #
    def __init__(self, min_range: Value, max_range: Value) -> None:

        #
        super().__init__()

        #
        self.min_range: Value = min_range
        self.max_range: Value = max_range

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        return float(
            random.randint(
                a=int(self.min_range.__getitem__(index=index, sample_rate=sample_rate)),
                b=int(self.max_range.__getitem__(index=index, sample_rate=sample_rate))
            )
        )

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        """
        Returns a vectorized array of random integers (as floats).
        """

        #
        ### Get the vectorized min and max boundaries. ###
        #
        min_vals: NDArray[np.float32] = self.min_range.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        max_vals: NDArray[np.float32] = self.max_range.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        ### Vectorized version using numpy's random generator. ###
        #
        min_int: NDArray[np.int64] = min_vals.astype(np.int64)

        #
        ### np.random.randint's 'high' parameter is exclusive, so we add 1 ###
        ### to match the inclusive behavior of random.randint.             ###
        #
        max_int: NDArray[np.int64] = max_vals.astype(np.int64) + 1

        #
        ### Ensure 'high' is always strictly greater than 'low'. ###
        ### If min_int >= max_int, set max_int to min_int + 1.   ###
        #
        max_int = np.maximum(min_int + 1, max_int)

        #
        ### Generate random ints and cast back to float32 for the audio buffer. ###
        #
        return np.random.randint(
            low=min_int,
            high=max_int,
            size=indexes_buffer.shape
        ).astype(np.float32)


#
class RandomFloat(Value):

    """
    A Value that returns a random float within a specified range
    for each sample.
    """

    #
    def __init__(self, min_range: Value, max_range: Value) -> None:

        #
        super().__init__()

        #
        self.min_range: Value = min_range
        self.max_range: Value = max_range

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        return random.uniform(
            a=float(self.min_range.__getitem__(index=index, sample_rate=sample_rate)),
            b=float(self.max_range.__getitem__(index=index, sample_rate=sample_rate))
        )

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        """
        Returns a vectorized array of random floats.
        This is a performance-critical override.
        """

        #
        ### Get the vectorized min and max boundaries. ###
        #
        min_vals: NDArray[np.float32] = self.min_range.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        max_vals: NDArray[np.float32] = self.max_range.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        ### Use numpy's vectorized uniform random number generator. ###
        #
        return np.random.uniform(
            low=min_vals,
            high=max_vals,
            size=indexes_buffer.shape
        ).astype(np.float32)


#
class RandomChoice(Value):

    """A Value that randomly selects from a list of other Value objects."""

    #
    def __init__(self, choices: list[Value]) -> None:

        #
        super().__init__()

        #
        self.choices: list[Value] = choices

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        return random.choice(self.choices).__getitem__(index=index, sample_rate=sample_rate)


#
### UTILS. ###
#

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


#
def c(v: float | int) -> Value:

    """Shorthand helper function to create a Constant Value."""

    #
    return Constant(value=v)


#
### BASIC OPERATION CLASSES ON SINGLE ITEMS. ###
#

#
class Polynom(Value):

    """
    A Value that calculates a polynomial function:
    y = terms[0] + terms[1]*X + terms[2]*X^2 + ...
    """

    #
    def __init__(
        self,
        X: Value,
        terms: list[Value] = [Constant(0), Constant(1)]
    ) -> None:

        #
        super().__init__()

        #
        self.X: Value = X
        #
        self.terms: list[Value] = terms

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        X_val: float = self.X.__getitem__(index=index, sample_rate=sample_rate)

        #
        return sum([
            X_val**i * self.terms[i].__getitem__(index=index, sample_rate=sample_rate)
            for i in range(len(self.terms))
        ])

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        X_val: NDArray[np.float32] = self.X.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        return np.sum([
            np.multiply( np.pow(X_val, i), self.terms[i].getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate) )
            for i in range(len(self.terms))
        ])


#
class BasicScaling(Value):

    """A Value that applies a linear transformation: value * mult_scale + sum_scale."""

    #
    def __init__(self, value: Value, mult_scale: Value, sum_scale: Value) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.mult_scale: Value = mult_scale
        self.sum_scale: Value = sum_scale

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        v: float = self.value.__getitem__(index=index, sample_rate=sample_rate)
        m: float = self.mult_scale.__getitem__(index=index, sample_rate=sample_rate)
        s: float = self.sum_scale.__getitem__(index=index, sample_rate=sample_rate)

        #
        return v * m + s

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        v: NDArray[np.float32] = self.value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        m: NDArray[np.float32] = self.mult_scale.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        s: NDArray[np.float32] = self.sum_scale.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        return np.multiply(v, m) + s


#
class Abs(Value):

    """A Value that returns the absolute value of another Value."""

    #
    def __init__(
        self,
        value: Value
    ) -> None:

        #
        super().__init__()

        #
        self.value: Value = value

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        return abs(self.value.__getitem__(index=index, sample_rate=sample_rate))

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        return np.abs(self.value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate))


#
class Clamp(Value):

    """
    A Value that constrains another Value between a min and max Value.
    Also known as Clip.
    """

    #
    def __init__(
        self,
        value: Value,
        min_value: Value = Constant(0),
        max_value: Value = Constant(1)
    ) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.min_value: Value = min_value
        self.max_value: Value = max_value

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        return max(
            self.min_value.__getitem__(index=index, sample_rate=sample_rate),
            min(
                self.max_value.__getitem__(index=index, sample_rate=sample_rate),
                self.value.__getitem__(index=index, sample_rate=sample_rate)
            )
        )

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        return np.clip(
            self.value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate),
            self.min_value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate),
            self.max_value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        )


#
class LowPass(Value):

    """
    A simple "clipper" Value that limits the maximum value.
    This is NOT an audio filter (like a Butterworth or RC filter).
    It is equivalent to min(value, max_value).
    """

    #
    def __init__(
        self,
        value: Value,
        max_value: Value = Constant(0)
    ) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.max_value: Value = max_value

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        return min(
            self.max_value.__getitem__(index=index, sample_rate=sample_rate),
            self.value.__getitem__(index=index, sample_rate=sample_rate)
        )

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        return np.minimum(
            self.max_value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate),
            self.value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        )


#
class HighPass(Value):

    """
    A simple "clipper" Value that limits the minimum value.
    This is NOT an audio filter (like a Butterworth or RC filter).
    It is equivalent to max(value, min_value).
    """

    #
    def __init__(
        self,
        value: Value,
        min_value: Value = Constant(0)
    ) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.min_value: Value = min_value

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        return max(
            self.min_value.__getitem__(index=index, sample_rate=sample_rate),
            self.value.__getitem__(index=index, sample_rate=sample_rate)
        )

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        return np.maximum(
            self.min_value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate),
            self.value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        )


#
class MaskTreshold(Value):

    """
    A Value that acts as a switch:
    if mask >= treshold, return mask_value.
    Otherwise, return the original value.
    """

    #
    def __init__(
        self,
        value: Value,
        mask: Value,
        treshold_to_mask: Value = Constant(1),
        mask_value: Value = Constant(0)
    ) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.mask: Value = mask
        self.treshold_to_mask: Value = treshold_to_mask
        self.mask_value: Value = mask_value

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        mask_v: float = self.mask.__getitem__(index=index, sample_rate=sample_rate)

        #
        if mask_v >= self.treshold_to_mask.__getitem__(index=index, sample_rate=sample_rate):

            #
            return self.mask_value.__getitem__(index=index, sample_rate=sample_rate)

        #
        else:

            #
            return self.value.__getitem__(index=index, sample_rate=sample_rate)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        base_value: NDArray[np.float32] = self.value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        masked_value: NDArray[np.float32] = self.mask_value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        mask_v: NDArray[np.float32] = self.mask.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        treshold_v: NDArray[np.float32] = self.treshold_to_mask.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        return np.where(mask_v < treshold_v, base_value, masked_value)


#
class TimeInterval(Value):

    """
    A Value that selects between two other Values based on the index (time).
    Returns `value_inside` if `min_sample_idx <= index <= max_sample_idx`.
    Otherwise, returns `value_outside`.
    """

    #
    def __init__(
        self,
        value_inside: Value,
        value_outside: Value = Constant(0),
        min_sample_idx: Value = Constant(0),
        max_sample_idx: Value = Constant(1)
    ) -> None:

        #
        super().__init__()

        #
        self.value_inside: Value = value_inside
        self.value_outside: Value = value_outside
        self.min_sample_idx: Value = min_sample_idx
        self.max_sample_idx: Value = max_sample_idx

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        if index < self.min_sample_idx.__getitem__(index=index, sample_rate=sample_rate):
            #
            return self.value_outside.__getitem__(index=index, sample_rate=sample_rate)

        #
        elif index > self.max_sample_idx.__getitem__(index=index, sample_rate=sample_rate):
            #
            return self.value_outside.__getitem__(index=index, sample_rate=sample_rate)

        #
        return self.value_inside.__getitem__(index=index, sample_rate=sample_rate)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        inside_values: NDArray[np.float32] = self.value_inside.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        outside_values: NDArray[np.float32] = self.value_outside.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        min_idx: NDArray[np.float32] = self.min_sample_idx.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        max_idx: NDArray[np.float32] = self.max_sample_idx.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        ### Create mask for values inside the interval. ###
        #
        inside_mask = (indexes_buffer >= min_idx) & (indexes_buffer <= max_idx)

        #
        return np.where(inside_mask, inside_values, outside_values)


#
### BASIC OPERATION CLASSES ON MULTIPLE ITEMS. ###
#

#
class Min(Value):

    """A Value that returns the minimum value from a list of input Values."""

    #
    def __init__(self, *values: Value | list[Value]) -> None:

        #
        super().__init__()

        #
        self.values: list[Value] = input_args_to_values(values=values)

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        return min([v.__getitem__(index=index, sample_rate=sample_rate) for v in self.values])

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        arrays = [v.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate) for v in self.values]
        #
        return np.minimum.reduce(arrays)


#
class Max(Value):

    """A Value that returns the maximum value from a list of input Values."""

    #
    def __init__(self, *values: Value | list[Value]) -> None:

        #
        super().__init__()

        #
        self.values: list[Value] = input_args_to_values(values=values)

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        return max([v.__getitem__(index=index, sample_rate=sample_rate) for v in self.values])

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        arrays = [v.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate) for v in self.values]
        #
        return np.maximum.reduce(arrays)


#
class Sum(Value):

    """A Value that returns the sum of a list of input Values."""

    #
    def __init__(self, *values: Value | list[Value]) -> None:

        #
        super().__init__()

        #
        self.values: list[Value] = input_args_to_values(values=values)

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        return sum([v.__getitem__(index=index, sample_rate=sample_rate) for v in self.values])

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        arrays = [v.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate) for v in self.values]
        #
        return np.sum(arrays, axis=0)


#
class PonderedSum(Value):

    """
    A Value that returns a weighted sum of (weight, value) pairs.
    Result = (weight1 * value1) + (weight2 * value2) + ...
    """

    #
    def __init__(self, values: list[tuple[Value, Value]]) -> None:

        #
        super().__init__()

        #
        self.values: list[tuple[Value, Value]] = values

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        result: float = 0

        #
        for pond, val in self.values:

            #
            result += pond.__getitem__(index=index, sample_rate=sample_rate) * val.__getitem__(index=index, sample_rate=sample_rate)

        #
        return result

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        result: NDArray[np.float32] = np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        for pond, val in self.values:

            #
            result += pond.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate) * val.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        return result


#
class Product(Value):

    """A Value that returns the product of a list of input Values."""

    #
    def __init__(self, *values: Value | list[Value]) -> None:

        #
        super().__init__()

        #
        self.values: list[Value] = input_args_to_values(values=values)

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        result: float = 1

        #
        for v in self.values:

            #
            result *= v.__getitem__(index=index, sample_rate=sample_rate)

        #
        return result

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        result: NDArray[np.float32] = np.ones_like(indexes_buffer, dtype=np.float32)

        #
        for v in self.values:

            #
            result = np.multiply(result, v.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate))

        #
        return result


#
### MORE COMPLEX MATHEMATICAL FUNCTIONS CLASSES. ###
#

#
class Pow(Value):

    """A Value that calculates base ^ exponent."""

    #
    def __init__(
        self,
        exponent: Value,
        base: Value = Constant(value=math.e)
    ) -> None:

        #
        super().__init__()

        #
        self.exponent: Value = exponent
        self.base: Value = base

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        base_v: float = self.base.__getitem__(index=index, sample_rate=sample_rate)
        exp_v: float = self.exponent.__getitem__(index=index, sample_rate=sample_rate)

        #
        return base_v ** exp_v

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        base_v: NDArray[np.float32] = self.base.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        exp_v: NDArray[np.float32] = self.exponent.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        return np.power(base_v, exp_v)


#
class Log(Value):

    """A Value that calculates log_base(value)."""

    #
    def __init__(
        self,
        value: Value,
        base: Value = Constant(value=math.e)
    ) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.base: Value = base

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        base_v: float = self.base.__getitem__(index=index, sample_rate=sample_rate)
        val_v: float = self.value.__getitem__(index=index, sample_rate=sample_rate)

        #
        return math.log(x=val_v, base=base_v)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        base_v: NDArray[np.float32] = self.base.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        val_v: NDArray[np.float32] = self.value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        return (np.log(val_v) / np.log(base_v)).astype(dtype=np.float32)


#
class Sin(Value):

    """
    A Value that generates a sine wave:
    amplitude * sin( (value * frequency) + delta )
    """

    #
    def __init__(
        self,
        value: Value,
        frequency: Value = Constant(1),
        amplitude: Value = Constant(1),
        delta: Value = Constant(0)
    ) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.frequency: Value = frequency
        self.amplitude: Value = amplitude
        self.delta: Value = delta

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        val_v: float = self.value.__getitem__(index=index, sample_rate=sample_rate)
        fre_v: float = self.frequency.__getitem__(index=index, sample_rate=sample_rate)
        amp_v: float = self.amplitude.__getitem__(index=index, sample_rate=sample_rate)
        del_v: float = self.delta.__getitem__(index=index, sample_rate=sample_rate)

        #
        return amp_v * math.sin( val_v * fre_v + del_v )

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        val_v: NDArray[np.float32] = self.value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        fre_v: NDArray[np.float32] = self.frequency.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        amp_v: NDArray[np.float32] = self.amplitude.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        del_v: NDArray[np.float32] = self.delta.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        return np.multiply( amp_v, np.sin( np.multiply(val_v, fre_v) + del_v ) )


#
class Cos(Value):

    """
    A Value that generates a cosine wave:
    amplitude * cos( (value * frequency) + delta )
    """

    #
    def __init__(
        self,
        value: Value,
        frequency: Value = Constant(1),
        amplitude: Value = Constant(1),
        delta: Value = Constant(0)
    ) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.frequency: Value = frequency
        self.amplitude: Value = amplitude
        self.delta: Value = delta

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        val_v: float = self.value.__getitem__(index=index, sample_rate=sample_rate)
        fre_v: float = self.frequency.__getitem__(index=index, sample_rate=sample_rate)
        amp_v: float = self.amplitude.__getitem__(index=index, sample_rate=sample_rate)
        del_v: float = self.delta.__getitem__(index=index, sample_rate=sample_rate)

        #
        return amp_v * math.cos( val_v * fre_v + del_v )

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        val_v: NDArray[np.float32] = self.value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        fre_v: NDArray[np.float32] = self.frequency.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        amp_v: NDArray[np.float32] = self.amplitude.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        del_v: NDArray[np.float32] = self.delta.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        return np.multiply( amp_v, np.cos( np.multiply(val_v, fre_v) + del_v ) )


#
class Triangle(Value):

    """
    A Value that generates a "naive" triangle wave.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": This is a mathematically correct, "naive" triangle
            wave. It is "truthful" in that sense.
        - "Good Listening": This implementation is **POOR** for "good listening"
            when used for audio-rate oscillators (e.g., frequencies > 20 Hz).
        - **Reason:** It produces strong aliasing (unwanted, inharmonic
            frequencies) because it has infinite sharp corners (harmonics)
            that are not band-limited. This aliasing sounds like a harsh,
            "digital" noise.
        - **Good Use:** This implementation is perfectly "realistic" and "good"
            for LFO (Low-Frequency Oscillator) use, where aliasing is not
            in the audible range.
    """

    #
    def __init__(
        self,
        value: Value,
        frequency: Value = Constant(1),
        amplitude: Value = Constant(1),
        delta: Value = Constant(0)
    ) -> None:

        """
        Initializes the Triangle oscillator.

        Args:
            value: The input phase Value (e.g., time).
            frequency: The frequency multiplier.
            amplitude: The amplitude (gain).
            delta: The phase offset.
        """

        #
        super().__init__()

        #
        self.value: Value = value
        self.frequency: Value = frequency
        self.amplitude: Value = amplitude
        self.delta: Value = delta

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        val_v: float = self.value.__getitem__(index=index, sample_rate=sample_rate)
        fre_v: float = self.frequency.__getitem__(index=index, sample_rate=sample_rate)
        amp_v: float = self.amplitude.__getitem__(index=index, sample_rate=sample_rate)
        del_v: float = self.delta.__getitem__(index=index, sample_rate=sample_rate)

        #
        ### Calculate the phase. ###
        #
        phase: float = val_v * fre_v + del_v

        #
        ### Triangle wave formula: 2 * |2 * (phase - floor(phase + 0.5))| - 1 ###
        ### This creates a wave that oscillates between -1 and 1              ###
        #
        triangle_value: float = 2.0 * abs(2.0 * (phase - math.floor(phase + 0.5))) - 1.0

        #
        ### Apply amplitude scaling. ###
        #
        return amp_v * triangle_value

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        val_v: NDArray[np.float32] = self.value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        fre_v: NDArray[np.float32] = self.frequency.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        amp_v: NDArray[np.float32] = self.amplitude.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        del_v: NDArray[np.float32] = self.delta.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        ### Calculate the phase. ###
        #
        phase: NDArray[np.float32] = np.multiply(val_v, fre_v) + del_v

        #
        ### Triangle wave formula: 2 * |2 * (phase - floor(phase + 0.5))| - 1 ###
        #
        triangle_value: NDArray[np.float32] = (2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0).astype(dtype=np.float32)

        #
        ### Apply amplitude scaling. ###
        #
        return np.multiply(amp_v, triangle_value)


#
class Square(Value):

    """
    A Value that generates a "naive" square wave with a variable duty cycle.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": This is a mathematically correct, "naive" square wave.
        - "Good Listening": This implementation is **VERY POOR** for
            "good listening" at audio rates.
        - **Reason:** It produces extremely strong aliasing due to the
            instantaneous vertical "jumps" (discontinuities) in the waveform.
            This will sound very harsh and noisy.
        - **Good Use:** This is perfect for LFOs, triggers, or gates.
    """

    #
    def __init__(
        self,
        value: Value,
        frequency: Value = Constant(1),
        amplitude: Value = Constant(1),
        delta: Value = Constant(0),
        duty_cycle: Value = Constant(0.5)
    ) -> None:

        """
        Initializes the Square oscillator.

        Args:
            value: The input phase Value (e.g., time).
            frequency: The frequency multiplier.
            amplitude: The amplitude (gain).
            delta: The phase offset.
            duty_cycle: The fraction of the period (0.0 to 1.0) for
                        which the signal is high.
        """

        #
        super().__init__()

        #
        self.value: Value = value
        self.frequency: Value = frequency
        self.amplitude: Value = amplitude
        self.delta: Value = delta
        self.duty_cycle: Value = duty_cycle

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        val_v: float = self.value.__getitem__(index=index, sample_rate=sample_rate)
        fre_v: float = self.frequency.__getitem__(index=index, sample_rate=sample_rate)
        amp_v: float = self.amplitude.__getitem__(index=index, sample_rate=sample_rate)
        del_v: float = self.delta.__getitem__(index=index, sample_rate=sample_rate)
        duty_v: float = self.duty_cycle.__getitem__(index=index, sample_rate=sample_rate)

        #
        ### Calculate the phase and normalize to [0, 1). ###
        #
        phase: float = val_v * fre_v + del_v
        normalized_phase: float = phase - math.floor(phase)

        #
        ### Square wave: high for duty_cycle portion, low for the rest. ###
        #
        if normalized_phase < duty_v:
            #
            return amp_v
        #
        else:
            #
            return -amp_v

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        val_v: NDArray[np.float32] = self.value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        fre_v: NDArray[np.float32] = self.frequency.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        amp_v: NDArray[np.float32] = self.amplitude.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        del_v: NDArray[np.float32] = self.delta.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        duty_v: NDArray[np.float32] = self.duty_cycle.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        ### Calculate the phase and normalize to [0, 1). ###
        #
        phase: NDArray[np.float32] = np.multiply(val_v, fre_v) + del_v
        normalized_phase: NDArray[np.float32] = phase - np.floor(phase)

        #
        ### Square wave: high for duty_cycle portion, low for the rest. ###
        #
        return np.where(normalized_phase < duty_v, amp_v, -amp_v)


#
class Sawtooth(Value):

    """
    A Value that generates a "naive" sawtooth wave.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": This is a mathematically correct, "naive" sawtooth wave.
        - "Good Listening": This implementation is **VERY POOR** for
            "good listening" at audio rates.
        - **Reason:** Like the square wave, this has an instantaneous
            discontinuity (the "drop") which causes massive aliasing.
        - **Good Use:** Excellent for LFOs.
        """

    #
    def __init__(
        self,
        value: Value,
        frequency: Value = Constant(1),
        amplitude: Value = Constant(1),
        delta: Value = Constant(0),
        direction: Value = Constant(1)  # 1 for rising, -1 for falling
    ) -> None:

        """
        Initializes the Sawtooth oscillator.

        Args:
            value: The input phase Value (e.g., time).
            frequency: The frequency multiplier.
            amplitude: The amplitude (gain).
            delta: The phase offset.
            direction: A Value (e.g., Constant(1) or Constant(-1)) that determines the slope.
                        >= 0 gives a rising sawtooth.
                        < 0 gives a falling sawtooth.
            """

        #
        super().__init__()

        #
        self.value: Value = value
        self.frequency: Value = frequency
        self.amplitude: Value = amplitude
        self.delta: Value = delta
        self.direction: Value = direction

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        val_v: float = self.value.__getitem__(index=index, sample_rate=sample_rate)
        fre_v: float = self.frequency.__getitem__(index=index, sample_rate=sample_rate)
        amp_v: float = self.amplitude.__getitem__(index=index, sample_rate=sample_rate)
        del_v: float = self.delta.__getitem__(index=index, sample_rate=sample_rate)
        dir_v: float = self.direction.__getitem__(index=index, sample_rate=sample_rate)

        #
        ### Calculate the phase and normalize to [0, 1). ###
        #
        phase: float = val_v * fre_v + del_v
        normalized_phase: float = phase - math.floor(phase)

        #
        ### Sawtooth wave: linear rise from -1 to 1 (or fall). ###
        #
        if dir_v >= 0:
            #
            ### Rising sawtooth: goes from -1 to 1. ###
            #
            sawtooth_value: float = 2.0 * normalized_phase - 1.0
        #
        else:
            #
            ### Falling sawtooth: goes from 1 to -1. ###
            #
            sawtooth_value: float = 1.0 - 2.0 * normalized_phase

        #
        ### Apply amplitude scaling. ###
        #
        return amp_v * sawtooth_value

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        val_v: NDArray[np.float32] = self.value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        fre_v: NDArray[np.float32] = self.frequency.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        amp_v: NDArray[np.float32] = self.amplitude.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        del_v: NDArray[np.float32] = self.delta.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        dir_v: NDArray[np.float32] = self.direction.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        ### Calculate the phase and normalize to [0, 1). ###
        #
        phase: NDArray[np.float32] = np.multiply(val_v, fre_v) + del_v
        normalized_phase: NDArray[np.float32] = phase - np.floor(phase)

        #
        ### Sawtooth wave: linear rise from -1 to 1 (or fall). ###
        #
        rising_sawtooth: NDArray[np.float32] = (2.0 * normalized_phase - 1.0).astype(dtype=np.float32)
        falling_sawtooth: NDArray[np.float32] = (1.0 - 2.0 * normalized_phase).astype(dtype=np.float32)
        sawtooth_value: NDArray[np.float32] = np.where(dir_v >= 0, rising_sawtooth, falling_sawtooth)

        #
        ### Apply amplitude scaling. ###
        #
        return np.multiply(amp_v, sawtooth_value)
