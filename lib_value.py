#
### Import Modules. ###
#
from typing import Iterable
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

    #
    def __init__(self) -> None:

        #
        pass

    #
    def __getitem__(self, index: int) -> float:

        #
        return 0

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        ### Not implemented, using non optimized placeholder. ###
        #
        default: NDArray[np.float32] = np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        for idx, i in enumerate(indexes_buffer):
            #
            default[idx] = self.__getitem__(index=int(i))

        #
        return default


#
### BASIC INSTANCES CLASSES. ###
#

#
class Constant(Value):

    #
    def __init__(self, value: float | int) -> None:

        #
        super().__init__()

        #
        self.value: float | int = value

    #
    def __getitem__(self, index: int) -> float:

        #
        return self.value

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        return np.full_like(indexes_buffer, fill_value=self.value, dtype=np.float32)



#
class Identity(Value):

    #
    def __init__(self) -> None:

        #
        super().__init__()

    #
    def __getitem__(self, index: int) -> float:

        #
        return float(index)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        return indexes_buffer


#
class RandomInt(Value):

    #
    def __init__(self, min_range: Value, max_range: Value) -> None:

        #
        super().__init__()

        #
        self.min_range: Value = min_range
        self.max_range: Value = max_range

    #
    def __getitem__(self, index: int) -> float:

        #
        return float(
            random.randint(
                a=int(self.min_range.__getitem__(index=index)),
                b=int(self.max_range.__getitem__(index=index))
            )
        )

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        min_vals: NDArray[np.float32] = self.min_range.getitem_np(indexes_buffer=indexes_buffer)
        max_vals: NDArray[np.float32] = self.max_range.getitem_np(indexes_buffer=indexes_buffer)

        #
        result: NDArray[np.float32] = np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        for i in range(len(indexes_buffer)):
            #
            result[i] = float(random.randint(a=int(min_vals[i]), b=int(max_vals[i])))

        #
        return result


#
class RandomFloat(Value):

    #
    def __init__(self, min_range: Value, max_range: Value) -> None:

        #
        super().__init__()

        #
        self.min_range: Value = min_range
        self.max_range: Value = max_range

    #
    def __getitem__(self, index: int) -> float:

        #
        return random.uniform(
            a=float(self.min_range.__getitem__(index=index)),
            b=float(self.max_range.__getitem__(index=index))
        )


#
class RandomChoice(Value):

    #
    def __init__(self, choices: list[Value]) -> None:

        #
        super().__init__()

        #
        self.choices: list[Value] = choices

    #
    def __getitem__(self, index: int) -> float:

        #
        return random.choice(self.choices).__getitem__(index=index)


#
### UTILS. ###
#

#
def input_args_to_values(values: Iterable[Value] | Iterable[Iterable[Value]]) -> Iterable[Value]:

    #
    if len(values) == 0:
        #
        return [Constant(value=0)]

    #
    if isinstance(values[0], Value):
        #
        return values

    #
    return values[0]


#
def c(v: float | int) -> Value:
    #
    return Constant(value=v)


#
### BASIC OPERATION CLASSES ON SINGLE ITEMS. ###
#

#
class Polynom(Value):

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
    def __getitem__(self, index: int) -> float:

        #
        X_val: float = self.X.__getitem__(index=index)

        #
        return sum([
            X_val**i * self.terms[i].__getitem__(index=index)
            for i in range(len(self.terms))
        ])

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        X_val: NDArray[np.float32] = self.X.getitem_np(indexes_buffer=indexes_buffer)

        #
        return sum([
            np.multiply( np.pow(X_val, i), self.terms[i].getitem_np(indexes_buffer=indexes_buffer) )
            for i in range(len(self.terms))
        ])



#
class BasicScaling(Value):

    #
    def __init__(self, value: Value, mult_scale: Value, sum_scale: Value) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.mult_scale: Value = mult_scale
        self.sum_scale: Value = sum_scale

    #
    def __getitem__(self, index: int) -> float:

        #
        v: float = self.value.__getitem__(index=index)
        m: float = self.mult_scale.__getitem__(index=index)
        s: float = self.sum_scale.__getitem__(index=index)

        #
        return v * m + s

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        v: NDArray[np.float32] = self.value.getitem_np(indexes_buffer=indexes_buffer)
        m: NDArray[np.float32] = self.mult_scale.getitem_np(indexes_buffer=indexes_buffer)
        s: NDArray[np.float32] = self.sum_scale.getitem_np(indexes_buffer=indexes_buffer)

        #
        return np.multiply(v, m) + s


#
class Abs(Value):

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
    def __getitem__(self, index: int) -> float:

        #
        return abs(self.value.__getitem__(index=index))

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        return np.abs(self.value.getitem_np(indexes_buffer=indexes_buffer))


#
class Clamp(Value):

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
    def __getitem__(self, index: int) -> float:

        #
        return max(
            self.min_value.__getitem__(index=index),
            min(
                self.max_value.__getitem__(index=index),
                self.value.__getitem__(index=index)
            )
        )

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        return np.clip(
            self.value.getitem_np(indexes_buffer=indexes_buffer),
            self.min_value.getitem_np(indexes_buffer=indexes_buffer),
            self.max_value.getitem_np(indexes_buffer=indexes_buffer)
        )


#
class LowPass(Value):

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
    def __getitem__(self, index: int) -> float:

        #
        return min(
            self.max_value.__getitem__(index=index),
            self.value.__getitem__(index=index)
        )

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        return np.minimum(
            self.max_value.getitem_np(indexes_buffer=indexes_buffer),
            self.value.getitem_np(indexes_buffer=indexes_buffer)
        )


#
class HighPass(Value):

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
    def __getitem__(self, index: int) -> float:

        #
        return np.maximum(
            self.min_value.getitem_np(indexes_buffer=indexes_buffer),
            self.value.getitem_np(indexes_buffer=indexes_buffer)
        )

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        return max(
            self.min_value.getitem_np(indexes_buffer=indexes_buffer),
            self.value.getitem_np(indexes_buffer=indexes_buffer)
        )


#
class MaskTreshold(Value):

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
    def __getitem__(self, index: int) -> float:

        #
        mask_v: float = self.mask.__getitem__(index=index)

        #
        if mask_v >= self.treshold_to_mask.__getitem__(index=index):

            #
            return self.mask_value.__getitem__(index=index)

        #
        else:

            #
            return self.value.__getitem__(index=index)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        base_value: NDArray[np.float32] = self.value.getitem_np(indexes_buffer=indexes_buffer)
        masked_value: NDArray[np.float32] = self.mask_value.getitem_np(indexes_buffer=indexes_buffer)

        #
        mask_v: NDArray[np.float32] = self.mask.getitem_np(indexes_buffer=indexes_buffer)
        treshold_v: NDArray[np.float32] = self.treshold_to_mask.getitem_np(indexes_buffer=indexes_buffer)

        #
        return np.where(mask_v < treshold_v, base_value, masked_value)


#
class TimeInterval(Value):

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
    def __getitem__(self, index: int) -> float:

        #
        if index < self.min_sample_idx:
            #
            return self.value_outside.__getitem__(index=index)

        #
        elif index > self.max_sample_idx:
            #
            return self.value_outside.__getitem__(index=index)

        #
        return self.value_inside.__getitem__(index=index)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        inside_values: NDArray[np.float32] = self.value_inside.getitem_np(indexes_buffer=indexes_buffer)
        outside_values: NDArray[np.float32] = self.value_outside.getitem_np(indexes_buffer=indexes_buffer)

        #
        min_idx: NDArray[np.float32] = self.min_sample_idx.getitem_np(indexes_buffer=indexes_buffer)
        max_idx: NDArray[np.float32] = self.max_sample_idx.getitem_np(indexes_buffer=indexes_buffer)

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

    #
    def __init__(self, *values: Iterable[Value] | Iterable[Iterable[Value]]) -> None:

        #
        super().__init__()

        #
        self.values: Iterable[Value] = input_args_to_values(values=values)

    #
    def __getitem__(self, index: int) -> float:

        #
        return min([v.__getitem__(index=index) for v in self.values])

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        arrays = [v.getitem_np(indexes_buffer=indexes_buffer) for v in self.values]
        #
        return np.minimum.reduce(arrays)

#
class Max(Value):

    #
    def __init__(self, *values: Iterable[Value] | Iterable[Iterable[Value]]) -> None:

        #
        super().__init__()

        #
        self.values: Iterable[Value] = input_args_to_values(values=values)

    #
    def __getitem__(self, index: int) -> float:

        #
        return max([v.__getitem__(index=index) for v in self.values])

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        arrays = [v.getitem_np(indexes_buffer=indexes_buffer) for v in self.values]
        #
        return np.maximum.reduce(arrays)


#
class Sum(Value):

    #
    def __init__(self, *values: Iterable[Value] | Iterable[Iterable[Value]]) -> None:

        #
        super().__init__()

        #
        self.values: Iterable[Value] = input_args_to_values(values=values)

    #
    def __getitem__(self, index: int) -> float:

        #
        return sum([v.__getitem__(index=index) for v in self.values])

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        arrays = [v.getitem_np(indexes_buffer=indexes_buffer) for v in self.values]
        #
        return np.sum(arrays, axis=0)


#
class PonderedSum(Value):

    #
    def __init__(self, values: list[tuple[Value, Value]]) -> None:

        #
        super().__init__()

        #
        self.values: list[tuple[Value, Value]] = values

    #
    def __getitem__(self, index: int) -> float:

        #
        result: float = 0

        #
        for pond, val in self.values:

            #
            result += pond.__getitem__(index=index) * val.__getitem__(index=index)

        #
        return result

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        result: NDArray[np.float32] = np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        for pond, val in self.values:

            #
            result += pond.getitem_np(indexes_buffer=indexes_buffer) * val.getitem_np(indexes_buffer=indexes_buffer)

        #
        return result


#
class Product(Value):

    #
    def __init__(self, *values: Iterable[Value] | Iterable[Iterable[Value]]) -> None:

        #
        super().__init__()

        #
        self.values: Iterable[Value] = input_args_to_values(values=values)

    #
    def __getitem__(self, index: int) -> float:

        #
        result: float = 1

        #
        for v in self.values:

            #
            result *= v.__getitem__(index=index)

        #
        return result

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        result: NDArray[np.float32] = np.ones_like(indexes_buffer, dtype=np.float32)

        #
        for v in self.values:

            #
            result = np.multiply(result, v.getitem_np(indexes_buffer=indexes_buffer))

        #
        return result


#
### MORE COMPLEX MATHEMATICAL FUNCTIONS CLASSES. ###
#

#
class Pow(Value):

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
    def __getitem__(self, index: int) -> float:

        #
        base_v: float = self.base.__getitem__(index=index)
        exp_v: float = self.exponent.__getitem__(index=index)

        #
        return base_v ** exp_v

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        base_v: NDArray[np.float32] = self.base.getitem_np(indexes_buffer=indexes_buffer)
        exp_v: NDArray[np.float32] = self.exponent.getitem_np(indexes_buffer=indexes_buffer)

        #
        return np.power(base_v, exp_v)


#
class Log(Value):

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
    def __getitem__(self, index: int) -> float:

        #
        base_v: float = self.base.__getitem__(index=index)
        val_v: float = self.value.__getitem__(index=index)

        #
        return math.log(x=val_v, base=base_v)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        base_v: NDArray[np.float32] = self.base.getitem_np(indexes_buffer=indexes_buffer)
        val_v: NDArray[np.float32] = self.value.getitem_np(indexes_buffer=indexes_buffer)

        #
        return np.log(val_v) / np.log(base_v)


#
class Sin(Value):

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
    def __getitem__(self, index: int) -> float:

        #
        val_v: float = self.value.__getitem__(index=index)
        fre_v: float = self.frequency.__getitem__(index=index)
        amp_v: float = self.amplitude.__getitem__(index=index)
        del_v: float = self.delta.__getitem__(index=index)

        #
        return amp_v * math.sin( val_v * fre_v + del_v )

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        val_v: NDArray[np.float32] = self.value.getitem_np(indexes_buffer=indexes_buffer)
        fre_v: NDArray[np.float32] = self.frequency.getitem_np(indexes_buffer=indexes_buffer)
        amp_v: NDArray[np.float32] = self.amplitude.getitem_np(indexes_buffer=indexes_buffer)
        del_v: NDArray[np.float32] = self.delta.getitem_np(indexes_buffer=indexes_buffer)

        #
        return np.multiply( amp_v, np.sin( np.multiply(val_v, fre_v), del_v ) )


#
class Cos(Value):

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
    def __getitem__(self, index: int) -> float:

        #
        val_v: float = self.value.__getitem__(index=index)
        fre_v: float = self.frequency.__getitem__(index=index)
        amp_v: float = self.amplitude.__getitem__(index=index)
        del_v: float = self.delta.__getitem__(index=index)

        #
        return amp_v * math.cos( val_v * fre_v + del_v )

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        val_v: NDArray[np.float32] = self.value.getitem_np(indexes_buffer=indexes_buffer)
        fre_v: NDArray[np.float32] = self.frequency.getitem_np(indexes_buffer=indexes_buffer)
        amp_v: NDArray[np.float32] = self.amplitude.getitem_np(indexes_buffer=indexes_buffer)
        del_v: NDArray[np.float32] = self.delta.getitem_np(indexes_buffer=indexes_buffer)

        #
        return np.multiply( amp_v, np.cos( np.multiply(val_v, fre_v), del_v ) )


#
class Triangle(Value):

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
    def __getitem__(self, index: int) -> float:

        #
        val_v: float = self.value.__getitem__(index=index)
        fre_v: float = self.frequency.__getitem__(index=index)
        amp_v: float = self.amplitude.__getitem__(index=index)
        del_v: float = self.delta.__getitem__(index=index)

        #
        ### Calculate the phase. ###
        #
        phase: float = val_v * fre_v + del_v

        #
        ### Triangle wave formula: 2 * |2 * (phase - floor(phase + 0.5))| - 1 ###
        ### This creates a wave that oscillates between -1 and 1              ###
        #
        triangle_value: float = 2 * abs(2 * (phase - math.floor(phase + 0.5))) - 1

        #
        ### Apply amplitude scaling. ###
        #
        return amp_v * triangle_value

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        val_v: NDArray[np.float32] = self.value.getitem_np(indexes_buffer=indexes_buffer)
        fre_v: NDArray[np.float32] = self.frequency.getitem_np(indexes_buffer=indexes_buffer)
        amp_v: NDArray[np.float32] = self.amplitude.getitem_np(indexes_buffer=indexes_buffer)
        del_v: NDArray[np.float32] = self.delta.getitem_np(indexes_buffer=indexes_buffer)

        #
        ### Calculate the phase. ###
        #
        phase: NDArray[np.float32] = np.multiply(val_v, fre_v) + del_v

        #
        ### Triangle wave formula: 2 * |2 * (phase - floor(phase + 0.5))| - 1 ###
        #
        triangle_value: NDArray[np.float32] = 2 * np.abs(2 * (phase - np.floor(phase + 0.5))) - 1

        #
        ### Apply amplitude scaling. ###
        #
        return np.multiply(amp_v, triangle_value)


#
class Square(Value):

    #
    def __init__(
        self,
        value: Value,
        frequency: Value = Constant(1),
        amplitude: Value = Constant(1),
        delta: Value = Constant(0),
        duty_cycle: Value = Constant(0.5)
    ) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.frequency: Value = frequency
        self.amplitude: Value = amplitude
        self.delta: Value = delta
        self.duty_cycle: Value = duty_cycle

    #
    def __getitem__(self, index: int) -> float:

        #
        val_v: float = self.value.__getitem__(index=index)
        fre_v: float = self.frequency.__getitem__(index=index)
        amp_v: float = self.amplitude.__getitem__(index=index)
        del_v: float = self.delta.__getitem__(index=index)
        duty_v: float = self.duty_cycle.__getitem__(index=index)

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
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        val_v: NDArray[np.float32] = self.value.getitem_np(indexes_buffer=indexes_buffer)
        fre_v: NDArray[np.float32] = self.frequency.getitem_np(indexes_buffer=indexes_buffer)
        amp_v: NDArray[np.float32] = self.amplitude.getitem_np(indexes_buffer=indexes_buffer)
        del_v: NDArray[np.float32] = self.delta.getitem_np(indexes_buffer=indexes_buffer)
        duty_v: NDArray[np.float32] = self.duty_cycle.getitem_np(indexes_buffer=indexes_buffer)

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

    """Sawtooth wave generator. Rises linearly and then drops sharply."""

    #
    def __init__(
        self,
        value: Value,
        frequency: Value = Constant(1),
        amplitude: Value = Constant(1),
        delta: Value = Constant(0),
        direction: Value = Constant(1)  # 1 for rising sawtooth, -1 for falling sawtooth
    ) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.frequency: Value = frequency
        self.amplitude: Value = amplitude
        self.delta: Value = delta
        self.direction: Value = direction

    #
    def __getitem__(self, index: int) -> float:

        #
        val_v: float = self.value.__getitem__(index=index)
        fre_v: float = self.frequency.__getitem__(index=index)
        amp_v: float = self.amplitude.__getitem__(index=index)
        del_v: float = self.delta.__getitem__(index=index)
        dir_v: float = self.direction.__getitem__(index=index)

        #
        ### Calculate the phase and normalize to [0, 1). ###
        #
        phase: float = val_v * fre_v + del_v
        normalized_phase: float = phase - math.floor(phase)

        #
        ### Sawtooth wave: linear rise from -1 to 1 (or fall if direction is -1). ###
        #
        if dir_v >= 0:
            #
            ### Rising sawtooth: goes from -1 to 1. ###
            #
            sawtooth_value: float = 2 * normalized_phase - 1
        #
        else:
            #
            ### Falling sawtooth: goes from 1 to -1. ###
            #
            sawtooth_value: float = 1 - 2 * normalized_phase

        #
        ### Apply amplitude scaling. ###
        #
        return amp_v * sawtooth_value

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        val_v: NDArray[np.float32] = self.value.getitem_np(indexes_buffer=indexes_buffer)
        fre_v: NDArray[np.float32] = self.frequency.getitem_np(indexes_buffer=indexes_buffer)
        amp_v: NDArray[np.float32] = self.amplitude.getitem_np(indexes_buffer=indexes_buffer)
        del_v: NDArray[np.float32] = self.delta.getitem_np(indexes_buffer=indexes_buffer)
        dir_v: NDArray[np.float32] = self.direction.getitem_np(indexes_buffer=indexes_buffer)

        #
        ### Calculate the phase and normalize to [0, 1). ###
        #
        phase: NDArray[np.float32] = np.multiply(val_v, fre_v) + del_v
        normalized_phase: NDArray[np.float32] = phase - np.floor(phase)

        #
        ### Sawtooth wave: linear rise from -1 to 1 (or fall if direction is -1). ###
        #
        rising_sawtooth: NDArray[np.float32] = 2 * normalized_phase - 1
        falling_sawtooth: NDArray[np.float32] = 1 - 2 * normalized_phase
        sawtooth_value: NDArray[np.float32] = np.where(dir_v >= 0, rising_sawtooth, falling_sawtooth)

        #
        ### Apply amplitude scaling. ###
        #
        return np.multiply(amp_v, sawtooth_value)
