#
### Import Modules. ###
#
from typing import cast, Callable, Any
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
class WhiteNoise(Value):

    #
    def __init__(self, seed: int = 42, scale: float = 1.0):

        #
        super().__init__()

        #
        self.seed: int = seed
        #
        self.scale: float = scale

    #
    ### Helper function for vectorized deterministic noise ###
    #
    @staticmethod
    def vectorized_noise(
        indexes_buffer: NDArray[np.float32],
        seed: int,
        scale: float
    ) -> NDArray[np.float32]:

        """
        Generates a deterministic, pseudo-random noise value for each index.

        This replaces the non-vectorizable, non-performant `hash()`-based
        noise in the original classes. This uses a simple, fast LCG (Linear
        Congruential Generator) which is "hash-like" and deterministic.

        Args:
            indexes_buffer: The buffer of sample indices.
            seed: An integer to vary the noise (e.g., 8191, 7919).
            scale: The final scaling factor (e.g., 1/5000.0).

        Returns:
            A NumPy array of noise values, one for each index.
        """

        #
        ### A simple LCG: (a * x + c) % m              ###
        ### We use bitwise-AND for a fast modulo 2^32. ###
        #
        idx_int: NDArray[np.uint32] = indexes_buffer.astype(np.uint32)
        noise_int: NDArray[np.uint32] = ((idx_int * seed + 12345) & 0xFFFFFFFF).astype(dtype=np.uint32)

        #
        ### Convert to float in range [-0.5, 0.5] ###
        #
        noise_float: NDArray[np.float32] = ((noise_int.astype(np.float32) / 0xFFFFFFFF) - 0.5).astype(dtype=np.float32)

        #
        ### Scale to match original intent (e.g., approx -50 to 50, then / 5000) ###
        #
        return (noise_float * 100.0 * scale).astype(dtype=np.float32)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        #
        return self.__class__.vectorized_noise(indexes_buffer, self.seed, self.scale)


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
def generate_harmonics(
    time: Value,
    base_frequency: float,
    num_harmonics: int,
    amplitude_falloff: float,
    sample_rate: int,
    base_amplitude: Value = Constant(1.0)
) -> Value:

    """
    Utility to generate a band-limited sum of harmonic sine waves.

    This function is crucial for "good listening" as it prevents
    aliasing by checking harmonics against the Nyquist frequency.

    Args:
        time: The base `Value` for time (e.g., from `song.render`).
        base_frequency: The fundamental frequency in Hz (e.g., 440.0).
        num_harmonics: The maximum number of harmonics to generate.
        amplitude_falloff: A multiplier for each successive harmonic's
            amplitude. (e.g., 0.5 means each harmonic is half the
            amplitude of the previous one).
        sample_rate: The audio sample rate (e.g., 44100).
        base_amplitude: A `Value` for the fundamental's amplitude.

    Returns:
        A `Sum` Value object containing all valid, band-limited harmonics.
    """

    #
    harmonics_list: list[Value] = []
    nyquist_limit: float = sample_rate / 2.0
    pi2: float = 2 * math.pi
    current_amplitude_multiplier: float = 1.0

    #
    for n in range(1, num_harmonics + 1):

        #
        ### Calculate the frequency of the Nth harmonic. ###
        #
        harmonic_freq_hz: float = base_frequency * n

        #
        ### This is the anti-aliasing check. ###
        #
        if harmonic_freq_hz >= nyquist_limit:
            #
            break  # Stop adding harmonics that are too high.

        #
        ### Calculate the amplitude for this harmonic. ###
        #
        amp_value: Value = Product(base_amplitude, Constant(current_amplitude_multiplier))

        #
        ### Add the new Sin wave to our list. ###
        #
        harmonics_list.append(
            Sin(
                value=time,
                frequency=Constant(harmonic_freq_hz * pi2),
                amplitude=amp_value
            )
        )

        #
        ### Apply the falloff for the *next* harmonic. ###
        #
        current_amplitude_multiplier *= amplitude_falloff

    #
    ### If no harmonics were valid, return silence. ###
    #
    if not harmonics_list:
        #
        return Constant(0.0)

    #
    ### Return a single Value object that sums all harmonics. ###
    #
    return Sum(harmonics_list)


#
def LFO(
    time: Value,
    rate_hz: Value,
    waveform_class: Callable[..., Value],
    amplitude: Value = Constant(1.0),
    delta: Value = Constant(0.0)
) -> Value:

    """
    Utility to create a Low-Frequency Oscillator (LFO).

    This helper function simplifies LFO creation by abstracting the
    frequency unit inconsistency in the oscillator APIs.
    - `Sin` and `Cos` expect frequency in radians per second.

    - `Triangle`, `Square`, `Sawtooth` expect frequency in Hz.


    This function always takes `rate_hz` in Hz and automatically
    converts it to the correct unit for the given `waveform_class`.
    """

    #
    freq_val: Value

    #
    ### Check if the oscillator is Sin/Cos, which need rad/s. ###
    #
    if waveform_class in [Sin, Cos]:
        #
        ### Convert Hz to rad/s (Hz * 2 * pi). ###
        #
        freq_val = Product(rate_hz, Constant(2 * math.pi))
    #
    else:
        #
        ### Triangle, Square, etc., already use Hz. ###
        #
        freq_val = rate_hz

    #
    ### Return the instantiated oscillator class. ###
    #
    return waveform_class(
        value=time,
        frequency=freq_val,
        amplitude=amplitude,
        delta=delta
    )

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
class Modulo(Value):

    """
    A Value that computes the modulo (remainder) of another Value.
    Result = value % modulo_value

    This is ideal for creating looping LFOs (Low-Frequency Oscillators)
    by using a looping time value as the input to other oscillators
    like `Sawtooth` or `Triangle`.
    """

    #
    def __init__(
        self,
        value: Value,
        modulo_value: Value = Constant(1.0)
    ) -> None:

        #
        super().__init__()

        #
        self.value: Value = value
        self.modulo_value: Value = modulo_value

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        val_v: float = self.value.__getitem__(index=index, sample_rate=sample_rate)
        mod_v: float = self.modulo_value.__getitem__(index=index, sample_rate=sample_rate)

        #
        ### Handle division by zero. ###
        #
        if mod_v == 0:
            #
            return val_v

        #
        ### Use % operator for correct "wrapping" behavior. ###
        #
        return val_v % mod_v

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        val_v: NDArray[np.float32] = self.value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        mod_v: NDArray[np.float32] = self.modulo_value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        ### Use np.mod for vectorized modulo. ###
        ### We use np.where to prevent division by zero. ###
        #
        return np.where(
            mod_v == 0,
            val_v,
            np.mod(val_v, mod_v)
        )


#
class Sequencer(Value):

    """
    A Value container that generates a sequence of "notes" or "events".

    This class automates the process of summing multiple `Value` objects
    that are triggered at different times.

    It takes a list of data (e.g., a list of (start_time, frequency, duration)
    tuples) and a "factory" function. It calls the factory for each
    item in the data list to create a `Value` object, and then
    creates a single `Sum` of all the created objects.
    """

    #
    def __init__(
        self,
        time: Value,
        # The factory function must accept `time` as its first argument,
        # followed by the unpacked arguments from the data tuple.
        # e.g.: factory(time, freq, start, dur)
        instrument_factory: Callable[..., Value],
        note_data_list: list[tuple[Any, ...]]
    ) -> None:

        #
        super().__init__()

        #
        ### Build the list of all note/event Value objects. ###
        #
        notes: list[Value] = []
        #
        for note_data in note_data_list:
            #
            # Call the factory, e.g.:
            #   PianoNote(time, *note_data)
            # where note_data = (frequency, start_time, duration)
            #
            notes.append(instrument_factory(time, *note_data))

        #
        ### The sequencer's total output is simply the sum of all notes. ###
        #
        self.sum: Sum = Sum(notes)

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        ### Proxy the call to the internal Sum object. ###
        #
        return self.sum.__getitem__(index=index, sample_rate=sample_rate)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        ### Proxy the call to the internal Sum object. ###
        #
        return self.sum.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

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
class ExponentialADSR(Value):

    """
    A "truthful" one-shot Attack-Decay-Sustain-Release envelope
    with *exponential* curves.

    This is an improvement on ADSR2, allowing for non-linear (concave or
    convex) curves for the Attack, Decay, and Release stages.

    - A curve value of 1.0 is linear (identical to ADSR2).
    - A curve value > 1.0 is "convex" (starts slow, ends fast).
    - A curve value < 1.0 is "concave" (starts fast, ends slow).
    """

    #
    def __init__(
        self,
        time: Value,
        note_start: float,
        note_duration: float,
        attack_time: float = 0.05,
        decay_time: float = 0.1,
        sustain_level: float = 0.7,
        release_time: float = 0.2,
        attack_curve: float = 0.5,  # Concave (fast) start
        decay_curve: float = 2.0,   # Convex (natural) decay
        release_curve: float = 2.0  # Convex (natural) release
    ) -> None:

        #
        super().__init__()

        #
        self.time: Value = time
        self.note_start: float = note_start
        self.note_duration: float = note_duration
        self.attack_time: float = max(0.001, attack_time)  # Prevent div by zero
        self.decay_time: float = max(0.001, decay_time)
        self.sustain_level: float = sustain_level
        self.release_time: float = max(0.001, release_time)

        #
        self.attack_curve: float = attack_curve
        self.decay_curve: float = decay_curve
        self.release_curve: float = release_curve

        #
        ### Pre-calculate stage end times. ###
        #
        self.attack_end: float = self.attack_time
        self.decay_end: float = self.attack_time + self.decay_time
        self.sustain_end: float = self.note_duration  # "note off" event
        self.release_end: float = self.note_duration + self.release_time

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        relative_time: float = t - self.note_start

        #
        ### Gate: if we are before the note or after the release, output 0. ###
        #
        if relative_time < 0 or relative_time > self.release_end:
            #
            return 0.0

        #
        ### Attack phase. ###
        #
        if relative_time < self.attack_end:
            #
            progress: float = relative_time / self.attack_time
            #
            return math.pow(progress, self.attack_curve)

        #
        ### Decay phase. ###
        #
        elif relative_time < self.decay_end:
            #
            progress: float = (relative_time - self.attack_time) / self.decay_time
            #
            return self.sustain_level + (1.0 - self.sustain_level) * math.pow(1.0 - progress, self.decay_curve)

        #
        ### Sustain phase. ###
        #
        elif relative_time < self.sustain_end:
            #
            return self.sustain_level

        #
        ### Release phase. ###
        #
        else:
            #
            progress: float = (relative_time - self.note_duration) / self.release_time
            #
            return self.sustain_level * math.pow(1.0 - progress, self.release_curve)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        #
        relative_time: NDArray[np.float32] = t - self.note_start

        #
        ### Gate: Create a mask for all samples inside the envelope's lifetime. ###
        #
        gate_mask: NDArray[np.bool_] = (
            (relative_time >= 0) & (relative_time <= self.release_end)
        )

        #
        if not np.any(gate_mask):
            #
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        ### Define the 4 stages and their values. ###
        #

        #
        ## ATTACK. ##
        #
        attack_mask: NDArray[np.bool_] = relative_time < self.attack_end
        attack_progress: NDArray[np.float32] = (relative_time / self.attack_time).astype(dtype=np.float32)
        #
        ## Ensure base is not negative (which happens if relative_time < 0). ##
        #
        attack_base = np.maximum(0.0, attack_progress)
        attack_val: NDArray[np.float32] = np.power(attack_base, self.attack_curve)

        #
        ## DECAY. ##
        #
        decay_mask: NDArray[np.bool_] = relative_time < self.decay_end
        decay_progress: NDArray[np.float32] = ((relative_time - self.attack_time) / self.decay_time).astype(dtype=np.float32)
        #
        ## Ensure base is not negative (which happens if progress > 1.0). ##
        #
        decay_base = np.maximum(0.0, 1.0 - decay_progress)
        decay_val: NDArray[np.float32] = (self.sustain_level + (1.0 - self.sustain_level) * np.power(decay_base, self.decay_curve)).astype(dtype=np.float32)

        #
        ## SUSTAIN. ##
        #
        sustain_mask: NDArray[np.bool_] = relative_time < self.sustain_end
        sustain_val: NDArray[np.float32] = np.full_like(relative_time, self.sustain_level)

        #
        ## RELEASE. ##
        #
        release_progress: NDArray[np.float32] = ((relative_time - self.note_duration) / self.release_time).astype(dtype=np.float32)
        #
        ## Ensure base is not negative. ##
        #
        release_base = np.maximum(0.0, 1.0 - release_progress)
        release_val: NDArray[np.float32] = (self.sustain_level * np.power(release_base, self.release_curve)).astype(dtype=np.float32)

        #
        ### Build the envelope with nested np.where. ###
        #
        env: NDArray[np.float32] = np.where(
            attack_mask,
            attack_val,
            np.where(
                decay_mask,
                decay_val,
                np.where(
                    sustain_mask,
                    sustain_val,
                    release_val  # The final 'else' case
                )
            )
        )

        #
        ### Apply the main gate mask to ensure output is 0 outside the envelope. ###
        #
        return env * gate_mask


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


#
class ExponentialDecay(Value):

    """
    A simple, one-shot exponential decay envelope.
    Perfect for percussion (Kick, Snare, HiHat).

    This is a "truthful" and "good listening" envelope.
    `env = exp(-relative_time * decay_rate)`
    """

    #
    def __init__(
        self,
        time: Value,
        start_time: float,
        decay_rate: float = 15.0  # e.g., 15 for snare, 8 for kick
    ) -> None:

        #
        super().__init__()

        #
        self.time: Value = time
        self.start_time: float = start_time
        self.decay_rate: float = decay_rate

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        relative_time: float = t - self.start_time

        #
        ### Gate: before the note, output 0. ###
        #
        if relative_time < 0:
            #
            return 0.0

        #
        ### Exponential decay. ###
        #
        return math.exp(-relative_time * self.decay_rate)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        #
        relative_time: NDArray[np.float32] = t - self.start_time

        #
        ### Gate: Create a mask for all samples at or after the start. ###
        #
        gate_mask: NDArray[np.bool_] = (relative_time >= 0)

        #
        if not np.any(gate_mask):
            #
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        safe_relative_time: NDArray[np.float32] = np.maximum(0.0, relative_time)

        #
        ### Calculate decay for all samples using the safe time. ###
        #
        decay_val: NDArray[np.float32] = np.exp(-safe_relative_time * self.decay_rate)

        #
        ### Apply gate mask to ensure output is 0 before the start. ###
        #
        return decay_val * gate_mask


#
class BandLimitedSawtooth(Value):

    """
    A "good listening" sawtooth wave built from a fixed number of harmonics.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": This is a "truthful" model of additive synthesis
            used to create a band-limited sawtooth wave.
        - "Good Listening": **GOOD**.
        - **Reason:** This class avoids the "naive" formula by summing
            `Sin` waves, following the model of the `WobbleBass` class
.
            This is a "fixed-harmonic-limit" oscillator.
        - **Compromise:** This is not *perfectly* band-limited (which
            would require checking `frequency * n` against `sample_rate`
            for every sample). Instead, it uses a fixed `num_harmonics`,
            which is a "good listening" compromise that is vectorizable
            and supports dynamic frequency (e.g., vibrato).
    """

    #
    def __init__(
        self,
        time: Value,
        frequency: Value,  # Frequency in Hz
        amplitude: Value = Constant(1.0),
        num_harmonics: int = 15
    ) -> None:

        #
        super().__init__()

        #
        self.time: Value = time
        self.frequency: Value = frequency
        self.amplitude: Value = amplitude
        self.num_harmonics: int = max(1, num_harmonics)
        self.pi2: float = 2 * math.pi

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t_v: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        f_v: float = self.frequency.__getitem__(index=index, sample_rate=sample_rate)
        a_v: float = self.amplitude.__getitem__(index=index, sample_rate=sample_rate)

        #
        output: float = 0.0
        #
        for n in range(1, self.num_harmonics + 1):
            #
            phase: float = t_v * f_v * n * self.pi2
            #
            output += (math.sin(phase) / n)

        #
        ### Normalize (approx. 2/pi) and apply amplitude. ###
        #
        return (output * 0.6366 * a_v)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        t_v: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        f_v: NDArray[np.float32] = self.frequency.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        a_v: NDArray[np.float32] = self.amplitude.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        output_array: NDArray[np.float32] = np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        ### Sum the harmonics. ###
        #
        for n in range(1, self.num_harmonics + 1):
            #
            phase: NDArray[np.float32] = (t_v * f_v * n * self.pi2).astype(dtype=np.float32)
            #
            output_array += (np.sin(phase) / n)

        #
        ### Normalize (approx. 2/pi) and apply amplitude. ###
        #
        return (output_array * 0.6366 * a_v).astype(dtype=np.float32)


#
class BandLimitedSquare(Value):

    """
    A "good listening" square wave built from a fixed number of harmonics.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": This is a "truthful" additive synthesis model
            of a band-limited square wave (which contains only odd harmonics).
        - "Good Listening": **GOOD**.
        - **Reason:** This avoids the "naive" formula and its massive
            aliasing by summing `Sin` waves. It uses the same "fixed-harmonic-limit"
            compromise as `BandLimitedSawtooth`.
    """

    #
    def __init__(
        self,
        time: Value,
        frequency: Value,  # Frequency in Hz
        amplitude: Value = Constant(1.0),
        num_harmonics: int = 15
    ) -> None:

        #
        super().__init__()

        #
        self.time: Value = time
        self.frequency: Value = frequency
        self.amplitude: Value = amplitude
        self.num_harmonics: int = max(1, num_harmonics)
        self.pi2: float = 2 * math.pi

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t_v: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        f_v: float = self.frequency.__getitem__(index=index, sample_rate=sample_rate)
        a_v: float = self.amplitude.__getitem__(index=index, sample_rate=sample_rate)

        #
        output: float = 0.0
        #
        for n in range(1, self.num_harmonics + 1):
            #
            harmonic: int = 2 * n - 1  # Only odd harmonics
            #
            phase: float = t_v * f_v * harmonic * self.pi2
            #
            output += (math.sin(phase) / harmonic)

        #
        ### Normalize (approx. 4/pi) and apply amplitude. ###
        #
        return (output * 0.7854 * a_v)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        t_v: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        f_v: NDArray[np.float32] = self.frequency.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        a_v: NDArray[np.float32] = self.amplitude.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        output_array: NDArray[np.float32] = np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        ### Sum the harmonics. ###
        #
        for n in range(1, self.num_harmonics + 1):
            #
            harmonic: int = 2 * n - 1  # Only odd harmonics
            #
            phase: NDArray[np.float32] = (t_v * f_v * harmonic * self.pi2).astype(dtype=np.float32)
            #
            output_array += (np.sin(phase) / harmonic)

        #
        ### Normalize (approx. 4/pi) and apply amplitude. ###
        #
        return (output_array * 0.7854 * a_v).astype(dtype=np.float32)


#
class ADSR2(Value):

    """
    A "truthful" one-shot Attack-Decay-Sustain-Release envelope.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": **EXCELLENT**. This is the "correct" model of an
            ADSR envelope. It's event-based, triggering on `note_start`.
        - "Good Listening": **GOOD**. This is a perfectly functional envelope.
        - **Improvement:** The "realism" could be slightly improved. This
            implementation uses linear (straight-line) ramps for A, D, and R.
            Real analog envelopes have *exponential* curves, which sound
            "snappier" and more "natural." However, linear is much simpler
            to implement and is a valid (and common) synthesis choice.
    """

    #
    def __init__(
        self,
        time: Value,
        note_start: float,
        note_duration: float,
        attack_time: float = 0.05,
        decay_time: float = 0.1,
        sustain_level: float = 0.7,
        release_time: float = 0.2
    ) -> None:

        #
        super().__init__()

        #
        self.time: Value = time
        self.note_start: float = note_start
        self.note_duration: float = note_duration
        self.attack_time: float = attack_time
        self.decay_time: float = decay_time
        self.sustain_level: float = sustain_level
        self.release_time: float = release_time

        #
        ### Pre-calculate stage end times for clarity. ###
        #
        self.attack_end: float = self.attack_time
        self.decay_end: float = self.attack_time + self.decay_time
        self.sustain_end: float = self.note_duration # This is the "note off" event
        self.release_end: float = self.note_duration + self.release_time

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        relative_time: float = t - self.note_start

        #
        ### Gate: if we are before the note or after the release, output 0. ###
        #
        if relative_time < 0 or relative_time > self.release_end:
            #
            return 0.0

        #
        ### Attack phase. ###
        #
        if relative_time < self.attack_end:
            #
            return relative_time / self.attack_time

        #
        ### Decay phase. ###
        #
        elif relative_time < self.decay_end:
            #
            decay_progress: float = (relative_time - self.attack_time) / self.decay_time
            #
            return 1.0 - (1.0 - self.sustain_level) * decay_progress

        #
        ### Sustain phase. ###
        #
        elif relative_time < self.sustain_end:
            #
            return self.sustain_level

        #
        ### Release phase. ###
        #
        else:
            #
            release_progress: float = (relative_time - self.note_duration) / self.release_time
            #
            return self.sustain_level * (1.0 - release_progress)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        #
        relative_time: NDArray[np.float32] = t - self.note_start

        #
        ### Gate: Create a mask for all samples inside the envelope's lifetime. ###
        #
        gate_mask: NDArray[np.bool_] = (
            (relative_time >= 0) & (relative_time <= self.release_end)
        )

        #
        if not np.any(gate_mask):
            #
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        ### Define the 4 stages and their values. ###
        #
        attack_mask: NDArray[np.bool_] = relative_time < self.attack_end
        attack_val: NDArray[np.float32] = relative_time / self.attack_time

        #
        decay_mask: NDArray[np.bool_] = relative_time < self.decay_end
        decay_progress: NDArray[np.float32] = (relative_time - self.attack_time) / self.decay_time
        decay_val: NDArray[np.float32] = (1.0 - (1.0 - self.sustain_level) * decay_progress).astype(dtype=np.float32)

        #
        sustain_mask: NDArray[np.bool_] = relative_time < self.sustain_end
        sustain_val: NDArray[np.float32] = np.full_like(relative_time, self.sustain_level)

        #
        ### The final 'else' is the release phase. ###
        #
        release_progress: NDArray[np.float32] = (relative_time - self.note_duration) / self.release_time
        release_val: NDArray[np.float32] = (self.sustain_level * (1.0 - release_progress)).astype(dtype=np.float32)

        #
        ### Build the envelope with nested np.where. ###
        #
        env: NDArray[np.float32] = np.where(
            attack_mask,
            attack_val,
            np.where(
                decay_mask,
                decay_val,
                np.where(
                    sustain_mask,
                    sustain_val,
                    release_val  # The final 'else' case
                )
            )
        )

        #
        ### Apply the main gate mask to ensure output is 0 outside the envelope. ###
        #
        return env * gate_mask


#
class Distortion(Value):

    """
    Guitar distortion effect using `tanh` soft clipping.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": **EXCELLENT**. This is a "truthful" and classic
            model of a waveshaper used for soft-clipping distortion.
        - "Good Listening": **GOOD**. It does exactly what it's supposed to do:
            adds harmonics by clipping the waveform. The `tanh` function
            provides a "warm" sound compared to "hard" clipping.
    """

    #
    def __init__(self, value: Value, drive: float = 5.0) -> None:

        """
        Initializes the distortion effect.

        Args:
            value: The input `Value` (the audio signal) to be distorted.
            drive: The amount of gain to apply before clipping.
                    Higher values = more distortion.
        """

        #
        super().__init__()

        #
        self.value: Value = value
        self.drive: float = drive

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        ### Apply gain (drive). ###
        #
        x: float = self.value.__getitem__(index=index, sample_rate=sample_rate) * self.drive

        #
        ### Soft clipping using tanh. ###
        #
        return math.tanh(x) * 0.5

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        ### Get the input signal buffer and apply gain (drive). ###
        #
        x: NDArray[np.float32] = (
            self.value.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate) * self.drive
        )

        #
        ### Apply vectorized soft clipping using np.tanh. ###
        #
        return (np.tanh(x) * 0.5).astype(dtype=np.float32)

