#
### Import Modules. ###
#
import math
#
import numpy as np
from numpy.typing import NDArray
#
import lib_value as lv


#
class GuitarString(lv.Value):

    """
    Simulates a plucked guitar string with Karplus-Strong-like characteristics.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": This is a "truthful" additive synthesis model. It models
            a fundamental, several harmonics, and a "brightness" parameter that
            correctly makes higher harmonics decay faster.
        - "Good Listening": This class is **POOR** for "good listening" at
            audio rates.
        - **Reason:** It uses naive `sin` calls for high-order harmonics
            (e.g., `frequency * 6`). If `frequency * 6` exceeds half the
            sample rate (Nyquist limit), it will cause significant **aliasing**,
            which sounds like a harsh, inharmonic "digital" noise.
        - **Improvement:** To be "realistic," this class would need to be
            band-limited, which requires knowing the `sample_rate`.
    """

    def __init__(
        self,
        time: lv.Value,
        frequency: float,
        start_time: float,
        duration: float = 3.0,
        brightness: float = 1.0
    ) -> None:

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.frequency: float = frequency
        self.start_time: float = start_time
        self.duration: float = duration
        self.brightness: float = brightness

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        relative_t: float = t - self.start_time

        #
        ### Gate: stop sound outside the note's duration. ###
        #
        if relative_t < 0 or relative_t > self.duration:
            #
            return 0.0

        #
        ### Envelope: sharp 0.002s attack, then exponential decay. ###
        #
        if relative_t < 0.002:
            #
            env: float = relative_t / 0.002
        #
        else:
            #
            env: float = math.exp(-relative_t * 1.2)

        #
        ### Fundamental and harmonics (additive synthesis) ###
        #
        fundamental: float = math.sin(2 * math.pi * self.frequency * relative_t)
        harmonic2: float = 0.6 * math.sin(2 * math.pi * self.frequency * 2 * relative_t)
        harmonic3: float = 0.4 * math.sin(2 * math.pi * self.frequency * 3 * relative_t)
        harmonic4: float = 0.25 * math.sin(2 * math.pi * self.frequency * 4 * relative_t)
        harmonic5: float = 0.15 * math.sin(2 * math.pi * self.frequency * 5 * relative_t)
        harmonic6: float = 0.1 * math.sin(2 * math.pi * self.frequency * 6 * relative_t)

        #
        ### High harmonics decay faster (brightness parameter) ###
        #
        decay_factor: float = math.exp(-relative_t * 2 * self.brightness)

        #
        ### Sum the signal. ###
        #
        signal: float = (
            fundamental +
            harmonic2 * (0.5 + 0.5 * decay_factor) +
            harmonic3 * decay_factor +
            harmonic4 * decay_factor * 0.8 +
            harmonic5 * decay_factor * 0.6 +
            harmonic6 * decay_factor * 0.4
        )

        #
        return env * signal * 0.25

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        ### Get time and relative time for the whole buffer. ###
        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        relative_t: NDArray[np.float32] = t - self.start_time

        #
        ### Gate mask: sound is only non-zero within the duration. ###
        #
        mask: NDArray[np.bool_] = (relative_t >= 0) & (relative_t <= self.duration)
        if not np.any(mask):
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        ### Envelope: sharp 0.002s attack, then exponential decay. ###
        #
        attack_env: NDArray[np.float32] = relative_t / 0.002
        decay_env: NDArray[np.float32] = np.exp(-relative_t * 1.2)
        env: NDArray[np.float32] = np.where(relative_t < 0.002, attack_env, decay_env)

        #
        ### Fundamental and harmonics (vectorized additive synthesis). ###
        #
        pi2: float = 2 * np.pi
        fundamental: NDArray[np.float32] = np.sin(pi2 * self.frequency * relative_t)
        harmonic2: NDArray[np.float32] = (0.6 * np.sin(pi2 * self.frequency * 2 * relative_t)).astype(dtype=np.float32)
        harmonic3: NDArray[np.float32] = (0.4 * np.sin(pi2 * self.frequency * 3 * relative_t)).astype(dtype=np.float32)
        harmonic4: NDArray[np.float32] = (0.25 * np.sin(pi2 * self.frequency * 4 * relative_t)).astype(dtype=np.float32)
        harmonic5: NDArray[np.float32] = (0.15 * np.sin(pi2 * self.frequency * 5 * relative_t)).astype(dtype=np.float32)
        harmonic6: NDArray[np.float32] = (0.1 * np.sin(pi2 * self.frequency * 6 * relative_t)).astype(dtype=np.float32)

        #
        ### High harmonics decay faster. ###
        #
        decay_factor: NDArray[np.float32] = np.exp(-relative_t * 2 * self.brightness)

        #
        ### Sum the signal. ###
        #
        signal: NDArray[np.float32] = (
            fundamental +
            harmonic2 * (0.5 + 0.5 * decay_factor) +
            harmonic3 * decay_factor +
            harmonic4 * decay_factor * 0.8 +
            harmonic5 * decay_factor * 0.6 +
            harmonic6 * decay_factor * 0.4
        ).astype(dtype=np.float32)

        #
        ### Apply envelope, gain, and gate mask. ###
        #
        return (env * signal * 0.25 * mask).astype(dtype=np.float32)


#
class GuitarString2(lv.Value):

    """
    Simulates a guitar string with distortion and harmonics.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": Another additive synthesis model with a more complex
            ADR (Attack-Decay-Release) envelope.
        - "Good Listening": **POOR**.
        - **Reason:**
            1.  **Aliasing:** Same problem as `GuitarString` due to naive harmonics.
            2.  **Bad Noise:** The original `hash()`-based noise is not
                "good listening." It's not truly random and creates a
                periodic, digital artifact.
        - **Improvement:** This `getitem_np` version replaces the "bad"
            `hash()` noise with a proper, deterministic vectorized noise
            generator (`_vectorized_noise`). This fixes one of the "good
            listening" problems, but the aliasing remains.
    """

    def __init__(
        self,
        time: lv.Value,
        frequency: float,
        start_time: float,
        duration: float,
        amplitude: float = 0.4
    ) -> None:

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.frequency: float = frequency
        self.start_time: float = start_time
        self.duration: float = duration
        self.amplitude: float = amplitude

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        relative_time: float = t - self.start_time

        #
        if relative_time < 0 or relative_time > self.duration:
            #
            return 0.0

        #
        ### Envelope (Attack-Decay-Release). ###
        #

        #
        ## Attack. ###
        #
        if relative_time < 0.01:
            #
            env: float = relative_time / 0.01

        #
        ## Decay. ##
        #
        elif relative_time < self.duration - 0.1:
            #
            env: float = 1.0 * math.exp(-relative_time * 0.5)

        #
        ## Release. ###
        #
        else:
            #
            release: float = (relative_time - (self.duration - 0.1)) / 0.1
            env: float = (1.0 * math.exp(-relative_time * 0.5)) * (1.0 - release)

        #
        if env <= 0:
            #
            return 0.0

        #
        ### String vibration with harmonics. ###
        #
        fundamental: float = math.sin(2 * math.pi * self.frequency * t)
        harmonic2: float = 0.5 * math.sin(2 * math.pi * self.frequency * 2 * t)
        harmonic3: float = 0.3 * math.sin(2 * math.pi * self.frequency * 3 * t)

        #
        signal: float = fundamental + harmonic2 + harmonic3

        #
        ### Add some noise for realism (this is the "bad" hash-based noise). ###
        #
        noise: float = (hash((index * 12345) % 1000000) % 100 - 50) / 5000.0

        #
        return self.amplitude * env * (signal + noise)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        relative_time: NDArray[np.float32] = t - self.start_time

        #
        ### Gate mask. ###
        #
        mask: NDArray[np.bool_] = (relative_time >= 0) & (relative_time <= self.duration)
        #
        if not np.any(mask):
            #
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        ### Vectorized Envelope (Attack-Decay-Release). ###
        #
        attack_part: NDArray[np.float32] = relative_time / 0.01
        decay_part: NDArray[np.float32] = np.exp(-relative_time * 0.5)

        #
        release_phase: NDArray[np.float32] = (relative_time - (self.duration - 0.1)) / 0.1
        release_part: NDArray[np.float32] = (decay_part * (1.0 - release_phase)).astype(dtype=np.float32)

        #
        # Build envelope with nested `np.where`
        #
        env: NDArray[np.float32] = np.where(
            relative_time < 0.01,
            attack_part,
            np.where(relative_time < self.duration - 0.1, decay_part, release_part)
        )
        env_mask: NDArray[np.bool_] = env > 0

        #
        # String vibration with harmonics
        #
        pi2t: NDArray[np.float32] = 2 * np.pi * t
        fundamental: NDArray[np.float32] = np.sin(self.frequency * pi2t)
        harmonic2: NDArray[np.float32] = (0.5 * np.sin(self.frequency * 2 * pi2t)).astype(dtype=np.float32)
        harmonic3: NDArray[np.float32] = (0.3 * np.sin(self.frequency * 3 * pi2t)).astype(dtype=np.float32)

        signal: NDArray[np.float32] = (fundamental + harmonic2 + harmonic3).astype(dtype=np.float32)

        #
        # Vectorized deterministic noise (the "Improvement")
        #
        noise: NDArray[np.float32] = lv.WhiteNoise.vectorized_noise(
            indexes_buffer,
            seed=12345,
            scale=1/5000.0
        )

        #
        return (self.amplitude * env * (signal + noise) * mask * env_mask).astype(dtype=np.float32)


#
class AcousticString(lv.Value):
    """
    Simulates an acoustic guitar string with natural decay.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": A model combining a sharp attack, exponential decay,
            rich harmonics, and string noise. This is a "truthful" model.
        - "Good Listening": **POOR**.
        - **Reason:**
            1.  **Aliasing:** Same problem as `GuitarString` due to naive harmonics.
            2.  **Bad Noise:** Same `hash()`-based noise problem as `GuitarString2`.
        - **Improvement:** Replaced `hash()` noise with `_vectorized_noise`.
            The aliasing problem remains, as it's fundamental to this
            synthesis method without `sample_rate` context.
    """

    def __init__(
        self,
        time: lv.Value,
        frequency: float,
        pluck_time: float,
        amplitude: float = 0.3,
        decay_rate: float = 2.0
    ) -> None:

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.frequency: float = frequency
        self.pluck_time: float = pluck_time
        self.amplitude: float = amplitude
        self.decay_rate: float = decay_rate

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        relative_time: float = t - self.pluck_time

        #
        if relative_time < 0 or relative_time > 3.0:
            #
            return 0.0

        #
        ### Quick attack (pluck). ###
        #
        if relative_time < 0.005:
            #
            attack: float = relative_time / 0.005
        #
        else:
            #
            attack: float = 1.0

        #
        ### Natural exponential decay. ###
        #
        decay: float = math.exp(-relative_time * self.decay_rate)
        #
        env: float = attack * decay

        #
        if env < 0.001:
            #
            return 0.0

        #
        ### Rich harmonic content. ###
        #
        fundamental: float = math.sin(2 * math.pi * self.frequency * t)
        harmonic2: float = 0.6 * math.sin(2 * math.pi * self.frequency * 2 * t)
        harmonic3: float = 0.4 * math.sin(2 * math.pi * self.frequency * 3 * t)
        harmonic4: float = 0.25 * math.sin(2 * math.pi * self.frequency * 4 * t)
        harmonic5: float = 0.15 * math.sin(2 * math.pi * self.frequency * 5 * t)

        #
        ### Add subtle string noise ("bad" hash-based). ###
        #
        noise: float = (hash((index * 8191) % 1000000) % 100 - 50) / 8000.0

        #
        signal: float = fundamental + harmonic2 + harmonic3 + harmonic4 + harmonic5 + noise

        #
        return self.amplitude * env * signal

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        #
        relative_time: NDArray[np.float32] = t - self.pluck_time

        #
        ### Gate mask (hard-coded 3.0s duration). ###
        #
        mask: NDArray[np.bool_] = (relative_time >= 0) & (relative_time <= 3.0)
        #
        if not np.any(mask):
            #
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        ### Vectorized Envelope (Attack/Decay). ###
        #
        attack_part: NDArray[np.float32] = relative_time / 0.005
        #
        decay_part: NDArray[np.float32] = np.exp(-relative_time * self.decay_rate)

        #
        env: NDArray[np.float32] = np.where(relative_time < 0.005, attack_part, 1.0) * decay_part
        env_mask: NDArray[np.bool_] = env >= 0.001

        #
        ### Rich harmonic content. ###
        #
        pi2t: NDArray[np.float32] = 2 * np.pi * t
        fundamental: NDArray[np.float32] = np.sin(self.frequency * pi2t)
        harmonic2: NDArray[np.float32] = (0.6 * np.sin(self.frequency * 2 * pi2t)).astype(dtype=np.float32)
        harmonic3: NDArray[np.float32] = (0.4 * np.sin(self.frequency * 3 * pi2t)).astype(dtype=np.float32)
        harmonic4: NDArray[np.float32] = (0.25 * np.sin(self.frequency * 4 * pi2t)).astype(dtype=np.float32)
        harmonic5: NDArray[np.float32] = (0.15 * np.sin(self.frequency * 5 * pi2t)).astype(dtype=np.float32)

        #
        ### Vectorized deterministic noise (the "Improvement"). ###
        #
        noise: NDArray[np.float32] = lv.WhiteNoise.vectorized_noise(
            indexes_buffer,
            seed=8191,
            scale=1/8000.0
        )

        #
        signal: NDArray[np.float32] = (
            fundamental + harmonic2 + harmonic3 + harmonic4 + harmonic5 + noise
        ).astype(dtype=np.float32)

        #
        return (self.amplitude * env * signal * mask * env_mask).astype(dtype=np.float32)


#
class Fingerpicking(lv.Value):

    """
    Generates a fingerpicking pattern by summing multiple AcousticString instances.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": This is a "container" class. Its truthness is high,
            as this is how patterns are constructed: by summing individual notes.
        - "Good Listening": This class will inherit all the "bad listening"
            (aliasing) problems from the `AcouS-ticString` objects it creates.
    """

    #
    def __init__(
        self,
        time: lv.Value,
        bass_note: float,
        chord_notes: list[float],
        start_time: float,
        pattern_duration: float = 2.0
    ) -> None:

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.bass_note: float = bass_note
        self.chord_notes: list[float] = chord_notes
        self.start_time: float = start_time
        self.pattern_duration: float = pattern_duration
        self.strings: list[AcousticString] = []

        #
        ### Classic Travis picking pattern: bass, treble, bass, treble. ###
        #
        ### Bass on beat 1 and 3. ###
        #
        self.strings.append(AcousticString(time, bass_note, start_time, 0.35, 1.5))
        self.strings.append(AcousticString(time, bass_note, start_time + pattern_duration/2, 0.35, 1.5))

        #
        ### Treble strings on off-beats. ###
        #
        eighth: float = pattern_duration / 8
        #
        for i, note_idx in enumerate([0, 1, 2, 1, 0, 1, 2, 1]):
            #
            ### Off-beats. ###
            #
            if i % 2 == 1:
                #
                pluck_time: float = start_time + i * eighth
                note: float = self.chord_notes[note_idx % len(self.chord_notes)]
                self.strings.append(AcousticString(time, note, pluck_time, 0.25, 2.0))

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        return sum(s.__getitem__(index=index, sample_rate=sample_rate) for s in self.strings)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        ### Create an array of all string outputs. ###
        #
        all_strings: list[NDArray[np.float32]] = [
            s.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate) for s in self.strings
        ]

        #
        ### Sum them all together. ###
        #
        return np.sum(all_strings, axis=0)


#
class Strum(lv.Value):

    """
    Simulates a guitar strum (chord with slight time offset between strings).

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": This is a "container" class. Its "truthness" is high,
            as it correctly models a strum by playing notes with a slight delay.
        - "Good Listening": This class will inherit all the "bad listening"
            (aliasing) problems from the `GuitarString` objects it creates.
    """

    #
    def __init__(
        self,
        time: lv.Value,
        frequencies: list[float],
        start_time: float,
        duration: float = 2.5
    ) -> None:

        #
        super().__init__()

        #
        self.strings: list[GuitarString] = []

        #
        for i, freq in enumerate(frequencies):
            #
            ### Each string is plucked slightly after the previous one. ###
            #
            offset: float = i * 0.015
            #
            self.strings.append(GuitarString(time, freq, start_time + offset, duration))

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        return sum(string.__getitem__(index=index, sample_rate=sample_rate) for string in self.strings)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        ### Create an array of all string outputs. ###
        #
        all_strings: list[NDArray[np.float32]] = [
            s.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate) for s in self.strings
        ]

        #
        ### Sum them all together. ###
        #
        return np.sum(all_strings, axis=0)

#
class PianoNote(lv.Value):
    """
    Simulates a piano note with harmonics using `lib_value` components.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": This is a "truthful" additive synthesis model.
            It is also **well-designed** because it's built by composing
            other `lv.Value` objects (`ADSR`, `lv.Sin`, `lv.Constant`).
        - "Good Listening": **GOOD**.
        - **Reason:** This class uses `lv.Sin` for its harmonics. A pure
            sine wave (`np.sin`) does not have aliasing artifacts (unless
            its *fundamental* frequency is above the Nyquist limit, which
            is user error).
        - **Improvement:** This is already a good class. A "more realistic"
            model would have higher harmonics decay faster, but this is
            a great "good listening" implementation.
    """

    #
    def __init__(
        self,
        time: lv.Value,
        frequency: float,
        start_time: float,
        duration: float,
        amplitude: float = 0.3
    ) -> None:

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.frequency: float = frequency
        self.start_time: float = start_time
        self.duration: float = duration
        self.amplitude: float = amplitude

        #
        ### Create ADSR envelope. ###
        #
        self.envelope: lv.ADSR2 = lv.ADSR2(
            time=time,
            note_start=start_time,
            note_duration=duration,
            attack_time=0.02,
            decay_time=0.15,
            sustain_level=0.6,
            release_time=0.3
        )

        #
        ### Piano has strong fundamental and harmonics. ###
        #
        self.fundamental: lv.Sin = lv.Sin(
            value=time,
            frequency=lv.Constant(frequency * (2 * math.pi)), # Freq in rad/s
            amplitude=lv.Constant(1.0)
        )
        #
        self.harmonic2: lv.Sin = lv.Sin(
            value=time,
            frequency=lv.Constant(frequency * 2 * (2 * math.pi)),
            amplitude=lv.Constant(0.4)
        )
        #
        self.harmonic3: lv.Sin = lv.Sin(
            value=time,
            frequency=lv.Constant(frequency * 3 * (2 * math.pi)),
            amplitude=lv.Constant(0.2)
        )
        #
        self.harmonic4: lv.Sin = lv.Sin(
            value=time,
            frequency=lv.Constant(frequency * 4 * (2 * math.pi)),
            amplitude=lv.Constant(0.1)
        )

        #
        ### We can also pre-build the sum of harmonics as a `lv.Sum` value. ###
        #
        self.harmonic_sum: lv.Sum = lv.Sum(
            [self.fundamental, self.harmonic2, self.harmonic3, self.harmonic4]
        )

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        envelope_val: float = self.envelope.__getitem__(index=index, sample_rate=sample_rate)
        #
        if envelope_val == 0:
            #
            return 0.0

        #
        ### Sum harmonics. ###
        #
        harmonic_sum: float = (
            self.fundamental.__getitem__(index=index, sample_rate=sample_rate) +
            self.harmonic2.__getitem__(index=index, sample_rate=sample_rate) +
            self.harmonic3.__getitem__(index=index, sample_rate=sample_rate) +
            self.harmonic4.__getitem__(index=index, sample_rate=sample_rate)
        )

        #
        ### Alternatively, using the pre-built `lv.Sum`                      ###
        ### harmonic_sum: float = self.harmonic_sum.__getitem__(index=index, sample_rate=sample_rate) ###
        #
        return self.amplitude * envelope_val * harmonic_sum

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        ### Get the vectorized envelope. ###
        #
        envelope_val: NDArray[np.float32] = self.envelope.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        ### Optimization: if the whole envelope is 0, return zeros. ###
        #
        if not np.any(envelope_val):
            #
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        ### Get the vectorized sum of harmonics. ###
        #
        harmonic_sum: NDArray[np.float32] = self.harmonic_sum.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        return (self.amplitude * envelope_val * harmonic_sum).astype(dtype=np.float32)


#
class PianoNote2(lv.Value):
    """
    Simulates a piano note with harmonics and a custom envelope.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": An additive synthesis model with a complex, 4-stage
            piecewise envelope. The model is "truthful" in its intent.
        - "Good Listening": **POOR**.
        - **Reason:**
            1.  **Aliasing:** Same problem as `GuitarString` due to naive harmonics.
            2.  **Envelope Clicks:** The envelope is piecewise (built with `if`
                statements). This creates sharp "corners" at the stage
                transitions, which can cause audible "clicks." A "good"
                envelope would be smoothly interpolated.
    """

    #
    def __init__(
        self,
        time: lv.Value,
        frequency: float,
        start_time: float,
        duration: float = 2.0
    ) -> None:

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.frequency: float = frequency
        self.start_time: float = start_time
        self.duration: float = duration

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        #
        relative_t: float = t - self.start_time

        #
        if relative_t < 0 or relative_t > self.duration + 0.5:
            #
            return 0.0

        #
        ### Complex 4-stage Envelope. ###
        #
        if relative_t < 0.01:
            #
            env: float = relative_t / 0.01
        #
        elif relative_t < 0.1:
            #
            env: float = 1.0 - 0.3 * (relative_t - 0.01) / 0.09
        #
        elif relative_t < self.duration:
            #
            env: float = 0.7 * math.exp(-relative_t * 0.8)
        #
        else:
            #
            env: float = 0.7 * math.exp(-self.duration * 0.8) * (
                1.0 - (relative_t - self.duration) / 0.5
            )

        #
        ### Harmonics. ###
        #
        fundamental: float = math.sin(2 * math.pi * self.frequency * relative_t)
        harmonic2: float = 0.5 * math.sin(2 * math.pi * self.frequency * 2 * relative_t)
        harmonic3: float = 0.25 * math.sin(2 * math.pi * self.frequency * 3 * relative_t)
        harmonic4: float = 0.15 * math.sin(2 * math.pi * self.frequency * 4 * relative_t)
        harmonic5: float = 0.1 * math.sin(2 * math.pi * self.frequency * 5 * relative_t)

        #
        return env * (fundamental + harmonic2 + harmonic3 + harmonic4 + harmonic5) * 0.3

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        #
        relative_t: NDArray[np.float32] = t - self.start_time

        #
        ### Gate mask. ###
        #
        mask: NDArray[np.bool_] = (relative_t >= 0) & (relative_t <= self.duration + 0.5)
        #
        if not np.any(mask):
            #
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        ### Vectorized 4-stage Envelope. ###
        #
        attack_part: NDArray[np.float32] = relative_t / 0.01
        decay1_part: NDArray[np.float32] = (1.0 - 0.3 * (relative_t - 0.01) / 0.09).astype(dtype=np.float32)
        decay2_part: NDArray[np.float32] = (0.7 * np.exp(-relative_t * 0.8)).astype(dtype=np.float32)
        release_part: NDArray[np.float32] = 0.7 * np.exp(-self.duration * 0.8) * (
            1.0 - (relative_t - self.duration) / 0.5
        )

        #
        ### Build envelope with nested `np.where`. ###
        #
        env: NDArray[np.float32] = np.where(
            relative_t < 0.01,
            attack_part,
            np.where(
                relative_t < 0.1,
                decay1_part,
                np.where(relative_t < self.duration, decay2_part, release_part)
            )
        )

        #
        ### Harmonics. ###
        #
        pi2: float = 2 * np.pi
        fundamental: NDArray[np.float32] = np.sin(pi2 * self.frequency * relative_t)
        harmonic2: NDArray[np.float32] = (0.5 * np.sin(pi2 * self.frequency * 2 * relative_t)).astype(dtype=np.float32)
        harmonic3: NDArray[np.float32] = (0.25 * np.sin(pi2 * self.frequency * 3 * relative_t)).astype(dtype=np.float32)
        harmonic4: NDArray[np.float32] = (0.15 * np.sin(pi2 * self.frequency * 4 * relative_t)).astype(dtype=np.float32)
        harmonic5: NDArray[np.float32] = (0.1 * np.sin(pi2 * self.frequency * 5 * relative_t)).astype(dtype=np.float32)

        #
        signal: NDArray[np.float32] = (
            fundamental + harmonic2 + harmonic3 + harmonic4 + harmonic5
        ).astype(dtype=np.float32)

        #
        return (env * signal * 0.3 * mask).astype(dtype=np.float32)


#
class WobbleBass(lv.Value):

    """
    Dubstep-style wobble bass with LFO modulation.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": This is a **VERY truthful** model of a classic
            subtractive-synthesis wobble bass. It models a sawtooth wave,
            an LFO, a filter (by multiplying the signal by the LFO),
            and distortion (using `tanh` clipping).
        - "Good Listening": **EXCELLENT**.
        - **Reason:** This class is *naturally band-limited*. The sawtooth
            wave is built from only 7 harmonics. This is a "realistic" and
            "good listening" technique that actively *prevents* aliasing.
            The `tanh` soft-clipping is also a "good" way to add distortion.
            This class is a model for how to do synthesis well.
    """

    #
    def __init__(
        self,
        time: lv.Value,
        base_frequency: float,
        start_time: float,
        duration: float,
        wobble_rate: float = 4.0,
        amplitude: float = 0.4
    ) -> None:

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.base_frequency: float = base_frequency
        self.start_time: float = start_time
        self.duration: float = duration
        self.wobble_rate: float = wobble_rate
        self.amplitude: float = amplitude

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        #
        relative_time: float = t - self.start_time

        #
        if relative_time < 0 or relative_time > self.duration:
            #
            return 0.0

        #
        ### LFO (Low Frequency Oscillator) for wobble. ###
        #
        lfo: float = (math.sin(2 * math.pi * self.wobble_rate * t) + 1.0) / 2.0

        #
        ### Generate rich harmonics (band-limited sawtooth). ###
        #
        sawtooth: float = 0.0
        #
        for harmonic in range(1, 8):
            #
            sawtooth += math.sin(2 * math.pi * self.base_frequency * harmonic * t) / harmonic

        #
        ### Apply LFO as a filter cutoff simulation. ###
        #
        filtered: float = sawtooth * (0.3 + 0.7 * lfo)

        #
        ### Soft clipping for distortion. ###
        #
        filtered = math.tanh(filtered * 2.0)

        #
        return self.amplitude * filtered

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        #
        relative_time: NDArray[np.float32] = t - self.start_time

        #
        ### Gate mask. ###
        #
        mask: NDArray[np.bool_] = (relative_time >= 0) & (relative_time <= self.duration)
        #
        if not np.any(mask):
            #
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        ### LFO (Low Frequency Oscillator) for wobble. ###
        #
        lfo: NDArray[np.float32] = ((np.sin(2 * np.pi * self.wobble_rate * t) + 1.0) / 2.0).astype(dtype=np.float32)

        #
        ### Generate rich harmonics (band-limited sawtooth). ###
        #
        pi2t: NDArray[np.float32] = 2 * np.pi * t
        #
        sawtooth: NDArray[np.float32] = np.zeros_like(indexes_buffer, dtype=np.float32)
        #
        for harmonic in range(1, 8):
            #
            sawtooth += np.sin(self.base_frequency * harmonic * pi2t) / harmonic

        #
        ### Apply LFO as a filter cutoff simulation. ###
        #
        filtered: NDArray[np.float32] = (sawtooth * (0.3 + 0.7 * lfo)).astype(dtype=np.float32)

        #
        ### Soft clipping for distortion. ###
        #
        filtered = np.tanh(filtered * 2.0)

        #
        return self.amplitude * filtered * mask


#
class DeepBass(lv.Value):

    """
    Deep sub-bass to accompany the drums.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": This is a **VERY truthful** model. A sub-bass
            is just a pure sine wave with a volume envelope.
        - "Good Listening": **EXCELLENT**.
        - **Reason:** It's a pure sine wave. It cannot produce aliasing
            (unless the fundamental frequency itself is > Nyquist, which
            is user error). This is the "cleanest" sound possible.
    """

    #
    def __init__(
        self,
        time: lv.Value,
        frequency: float,
        start_time: float,
        duration: float = 0.5
    ) -> None:

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.frequency: float = frequency
        self.start_time: float = start_time
        self.duration: float = duration

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        #
        relative_t: float = t - self.start_time

        #
        if relative_t < 0 or relative_t > self.duration:
            #
            return 0.0

        #
        ### Exponential decay. ###
        #
        env: float = math.exp(-relative_t * 6)

        #
        ### Pure sine wave for sub-bass. ###
        #
        bass: float = math.sin(2 * math.pi * self.frequency * relative_t)

        #
        return env * bass * 0.4

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        #
        relative_t: NDArray[np.float32] = t - self.start_time

        #
        ### Gate mask. ###
        #
        mask: NDArray[np.bool_] = (relative_t >= 0) & (relative_t <= self.duration)
        #
        if not np.any(mask):
            #
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        ### Exponential decay. ###
        #
        env: NDArray[np.float32] = np.exp(-relative_t * 6)

        #
        ### Pure sine wave for sub-bass. ###
        #
        bass: NDArray[np.float32] = np.sin(2 * np.pi * self.frequency * relative_t)

        #
        return (env * bass * 0.4 * mask).astype(dtype=np.float32)


#
class SaxophoneNote(lv.Value):

    """
    Simulates a saxophone note with realistic timbre.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": This is a **VERY truthful** model. It includes
            vibrato (via phase modulation), a correct emphasis on odd
            harmonics, and "breath noise."
        - "Good Listening": **POOR**.
        - **Reason:**
            1.  **Aliasing:** Same problem as `GuitarString` due to naive harmonics.
            2.  **Bad Noise:** Same `hash()`-based noise problem as `GuitarString2`.
        - **Improvement:** Replaced `hash()` noise with `_vectorized_noise`.
            The vibrato (FM) and aliasing interaction will be complex and
            likely produce even *more* inharmonic aliasing.
    """

    #
    def __init__(
        self,
        time: lv.Value,
        frequency: float,
        start_time: float,
        duration: float,
        amplitude: float = 0.3
    ) -> None:

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.frequency: float = frequency
        self.start_time: float = start_time
        self.duration: float = duration
        self.amplitude: float = amplitude

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        #
        relative_time: float = t - self.start_time

        #
        if relative_time < 0 or relative_time > self.duration:
            #
            return 0.0

        #
        ### Saxophone envelope - smooth attack and release. ###
        #
        if relative_time < 0.1:
            #
            env: float = relative_time / 0.1
        #
        elif relative_time < self.duration - 0.15:
            #
            env: float = 1.0
        #
        else:
            #
            release: float = (relative_time - (self.duration - 0.15)) / 0.15
            env: float = 1.0 - release

        #
        if env <= 0:
            #
            return 0.0

        #
        ### Add vibrato for expressiveness. ###
        #
        vibrato_mod: float = 1.0 + 0.01 * math.sin(2 * math.pi * 5.5 * t)
        freq: float = self.frequency * vibrato_mod

        #
        ### Saxophone has strong odd harmonics. ###
        #
        fundamental: float = math.sin(2 * math.pi * freq * t)
        harmonic2: float = 0.3 * math.sin(2 * math.pi * freq * 2 * t)
        harmonic3: float = 0.6 * math.sin(2 * math.pi * freq * 3 * t)
        harmonic4: float = 0.15 * math.sin(2 * math.pi * freq * 4 * t)
        harmonic5: float = 0.4 * math.sin(2 * math.pi * freq * 5 * t)

        #
        signal: float = fundamental + harmonic2 + harmonic3 + harmonic4 + harmonic5

        #
        ### Add breath noise ("bad" hash-based). ###
        #
        breath: float = (hash((index * 7919) % 1000000) % 100 - 50) / 1000.0

        #
        return self.amplitude * env * (signal + breath * 0.5)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        relative_time: NDArray[np.float32] = t - self.start_time

        #
        ### Gate mask. ###
        #
        mask: NDArray[np.bool_] = (relative_time >= 0) & (relative_time <= self.duration)
        #
        if not np.any(mask):
            #
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        ### Vectorized Envelope (Attack-Sustain-Release). ###
        #
        attack_part: NDArray[np.float32] = relative_time / 0.1
        sustain_part: NDArray[np.float32] = np.ones_like(indexes_buffer, dtype=np.float32)
        release_phase: NDArray[np.float32] = (relative_time - (self.duration - 0.15)) / 0.15
        release_part: NDArray[np.float32] = 1.0 - release_phase

        #
        ### Build envelope with nested `np.where`. ###
        #
        env: NDArray[np.float32] = np.where(
            relative_time < 0.1,
            attack_part,
            np.where(relative_time < self.duration - 0.15, sustain_part, release_part)
        )
        #
        env_mask: NDArray[np.bool_] = env > 0

        #
        ### Add vibrato for expressiveness. ###
        #
        vibrato_mod: NDArray[np.float32] = (1.0 + 0.01 * np.sin(2 * np.pi * 5.5 * t)).astype(dtype=np.float32)
        freq: NDArray[np.float32] = self.frequency * vibrato_mod

        #
        ### Saxophone has strong odd harmonics. ###
        #
        pi2t: NDArray[np.float32] = 2 * np.pi * t
        fundamental: NDArray[np.float32] = np.sin(freq * pi2t)
        harmonic2: NDArray[np.float32] = (0.3 * np.sin(freq * 2 * pi2t)).astype(dtype=np.float32)
        harmonic3: NDArray[np.float32] = (0.6 * np.sin(freq * 3 * pi2t)).astype(dtype=np.float32)
        harmonic4: NDArray[np.float32] = (0.15 * np.sin(freq * 4 * pi2t)).astype(dtype=np.float32)
        harmonic5: NDArray[np.float32] = (0.4 * np.sin(freq * 5 * pi2t)).astype(dtype=np.float32)

        #
        signal: NDArray[np.float32] = (
            fundamental + harmonic2 + harmonic3 + harmonic4 + harmonic5
        ).astype(dtype=np.float32)

        #
        ### Vectorized deterministic noise (the "Improvement"). ###
        #
        breath: NDArray[np.float32] = lv.WhiteNoise.vectorized_noise(
            indexes_buffer,
            seed=7919,
            scale=1/1000.0
        )

        #
        return (self.amplitude * env * (signal + breath * 0.5) * mask * env_mask).astype(dtype=np.float32)
