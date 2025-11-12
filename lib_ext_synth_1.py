#
### Import Modules. ###
#
import lib_value as lv
import math
#
import numpy as np
from numpy.typing import NDArray


#
class SynthLead(lv.Value):

    """
    Synthesizer lead sound with vibrato and filter sweep.

    "Truthness" / "Good Listening" Analysis:
      - "Truthness": The *model* is "truthful" (Osc + Filter + Env + LFO).
        However, the *implementation* of the oscillator and filter is not.
      - "Good Listening": **VERY POOR**.
      - **Reason 1 (Aliasing):** The sawtooth wave is "naive"
        (using `phase % (2 * math.pi)`). This creates infinite
        harmonics and will produce *massive* aliasing, which
        sounds like harsh, inharmonic, "digital" noise.
      - **Reason 2 (Fake Filter):** The "Low-pass filter sweep" is
        **NOT a filter**. It is just a gain control / crossfade.
        It will not remove high frequencies and will not sound
        like a real filter sweep.
    """

    def __init__(
        self,
        time: lv.Value,
        frequency: float,
        start_time: float,
        duration: float = 1.0
    ) -> None:

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.frequency: float = frequency
        self.start_time: float = start_time
        self.duration: float = duration
        self.pi2: float = 2 * math.pi

        #
        ### ADSR parameters. ###
        #
        self.attack: float = 0.1
        self.decay: float = 0.15
        self.sustain_level: float = 0.7
        self.release: float = 0.2
        self.release_start: float = self.duration - self.release

        #
        ### Vibrato parameters. ###
        #
        self.vibrato_rate: float = 5.0
        self.vibrato_depth: float = 0.02

    #
    def __getitem__(self, index: int) -> float:

        #
        t: float = self.time.__getitem__(index=index)
        relative_t: float = t - self.start_time

        #
        ### Gate. ###
        #
        if relative_t < 0 or relative_t > self.duration:
            #
            return 0.0

        #
        ### ADSR envelope. ###
        #
        env: float
        #
        if relative_t < self.attack:
            #
            env = relative_t / self.attack
        #
        elif relative_t < self.attack + self.decay:
            #
            progress: float = (relative_t - self.attack) / self.decay
            env = 1.0 - (1.0 - self.sustain_level) * progress
        #
        elif relative_t < self.release_start:
            #
            env = self.sustain_level
        #
        else:
            #
            progress: float = (relative_t - self.release_start) / self.release
            env = self.sustain_level * (1.0 - progress)

        #
        ### Vibrato (frequency modulation). ###
        #
        vibrato: float = 1.0 + self.vibrato_depth * math.sin(
            self.pi2 * self.vibrato_rate * relative_t
        )

        #
        ### Sawtooth wave (rich in harmonics, typical synth sound) ###
        ### This is the "naive," aliasing version.                 ###
        #
        phase: float = self.pi2 * self.frequency * vibrato * relative_t
        normalized_phase: float = phase % self.pi2
        sawtooth: float = 2 * (normalized_phase / self.pi2) - 1

        #
        ### "Low-pass filter sweep" (This is a fake filter). ###
        #
        cutoff: float = 1.0 - 0.5 * math.exp(-relative_t * 3)
        filtered: float = sawtooth * cutoff + sawtooth * (1 - cutoff) * 0.1

        #
        return env * filtered * 0.3

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer)
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
        ### Vectorized ADSR envelope. ###
        #
        attack_val: NDArray[np.float32] = relative_t / self.attack

        #
        decay_progress: NDArray[np.float32] = (relative_t - self.attack) / self.decay
        decay_val: NDArray[np.float32] = 1.0 - (1.0 - self.sustain_level) * decay_progress

        #
        sustain_val: NDArray[np.float32] = np.full_like(relative_t, self.sustain_level)

        #
        release_progress: NDArray[np.float32] = (relative_t - self.release_start) / self.release
        release_val: NDArray[np.float32] = self.sustain_level * (1.0 - release_progress)

        #
        env: NDArray[np.float32] = np.where(
            relative_t < self.attack,
            attack_val,
            np.where(
                relative_t < self.attack + self.decay,
                decay_val,
                np.where(
                    relative_t < self.release_start,
                    sustain_val,
                    release_val
                )
            )
        )

        #
        ### Vibrato (frequency modulation). ###
        #
        vibrato: NDArray[np.float32] = (1.0 + self.vibrato_depth * np.sin(
            self.pi2 * self.vibrato_rate * relative_t
        )).astype(dtype=np.float32)

        #
        ### Naive sawtooth wave (vectorized, will alias). ###
        #
        phase: NDArray[np.float32] = self.pi2 * self.frequency * vibrato * relative_t
        normalized_phase: NDArray[np.float32] = np.mod(phase, self.pi2)
        sawtooth: NDArray[np.float32] = 2 * (normalized_phase / self.pi2) - 1

        #
        ### "Low-pass filter sweep" (The "fake" filter). ###
        #
        cutoff: NDArray[np.float32] = (1.0 - 0.5 * np.exp(-relative_t * 3)).astype(dtype=np.float32)
        filtered: NDArray[np.float32] = (sawtooth * cutoff + sawtooth * (1 - cutoff) * 0.1).astype(dtype=np.float32)

        #
        return env * filtered * 0.3 * mask


#
class SynthBass(lv.Value):

    """
    Deep synthesizer bass.

    "Truthness" / "Good Listening" Analysis:
      - "Truthness": **EXCELLENT**. This is a "truthful" model of a
        classic bass patch: a square wave mixed with a sine-wave
        sub-oscillator (one octave lower).
      - "Good Listening": **VERY POOR**.
      - **Reason (Aliasing):** The square wave is "naive"
        (using `math.sin(phase) > 0`). This creates infinite
        harmonics and will alias *massively*, sounding harsh,
        brittle, and "digital," not "fat."
      - The sub-oscillator (pure `sin`) is "good listening" and
        will not alias.
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
        self.pi2: float = 2 * math.pi

    #
    def __getitem__(self, index: int) -> float:

        #
        t: float = self.time.__getitem__(index=index)
        relative_t: float = t - self.start_time

        #
        if relative_t < 0 or relative_t > self.duration:
            #
            return 0.0

        #
        ### Quick attack, exponential decay. ###
        #
        env: float
        #
        if relative_t < 0.01:
            #
            env = relative_t / 0.01
        #
        else:
            #
            env = math.exp(-(relative_t - 0.01) * 5)

        #
        ### Square wave for fat bass sound ("naive," aliasing version). ###
        #
        phase: float = self.pi2 * self.frequency * relative_t
        square: float = 1.0 if math.sin(phase) > 0 else -1.0

        #
        ### Add sub-oscillator (sine wave one octave down). ###
        #
        sub: float = 0.4 * math.sin(self.pi2 * (self.frequency / 2.0) * relative_t)

        #
        return env * (square * 0.6 + sub) * 0.35

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer)
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
        ### Quick attack, exponential decay (vectorized). ###
        #
        attack_val: NDArray[np.float32] = relative_t / 0.01
        decay_val: NDArray[np.float32] = np.exp(-(relative_t - 0.01) * 5)
        env: NDArray[np.float32] = np.where(relative_t < 0.01, attack_val, decay_val)

        #
        ### Square wave ("naive," aliasing version). ###
        #
        phase: NDArray[np.float32] = self.pi2 * self.frequency * relative_t
        square: NDArray[np.float32] = np.where(np.sin(phase) > 0, 1.0, -1.0)

        #
        ### Add sub-oscillator (sine wave one octave down). ###
        #
        sub: NDArray[np.float32] = (0.4 * np.sin(
            self.pi2 * (self.frequency / 2.0) * relative_t
        )).astype(dtype=np.float32)

        #
        return (env * (square * 0.6 + sub) * 0.35 * mask).astype(dtype=np.float32)


#
class SynthPad(lv.Value):

    """
    Atmospheric synthesizer pad.

    "Truthness" / "Good Listening" Analysis:
      - "Truthness": **EXCELLENT**. This is a "truthful" and classic
        model of a "supersaw" or "chorus" pad. It uses a slow
        Attack-Sustain-Release envelope and three detuned oscillators
        to create a wide, shimmering, atmospheric sound.
      - "Good Listening": **EXCELLENT**.
      - **Reason:** This class is built *only* from sine waves (`math.sin`).
        Pure sine waves do not produce aliasing artifacts (unless
        their fundamental frequency is set too high by the user).
        This is a "realistic" and "good listening" implementation.
    """

    #
    def __init__(
        self,
        time: lv.Value,
        frequency: float,
        start_time: float,
        duration: float = 4.0
    ) -> None:

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.frequency: float = frequency
        self.start_time: float = start_time
        self.duration: float = duration
        self.pi2: float = 2 * math.pi

        #
        ### Envelope parameters. ###
        #
        self.attack: float = 0.5
        self.release: float = 1.0
        self.release_start: float = self.duration - self.release

        #
        ### Oscillator detune amounts. ###
        #
        self.detune1: float = 1.003
        self.detune2: float = 0.997

    #
    def __getitem__(self, index: int) -> float:

        #
        t: float = self.time.__getitem__(index=index)
        relative_t: float = t - self.start_time

        #
        if relative_t < 0 or relative_t > self.duration:
            #
            return 0.0

        #
        ### Very slow attack and release (ASR envelope). ###
        #
        env: float
        #
        if relative_t < self.attack:
            #
            env = relative_t / self.attack
        #
        elif relative_t < self.release_start:
            #
            env = 1.0
        #
        else:
            #
            progress: float = (relative_t - self.release_start) / self.release
            env = 1.0 - progress

        #
        ### Multiple detuned oscillators for chorus effect. ###
        #
        osc1: float = math.sin(self.pi2 * self.frequency * relative_t)
        osc2: float = math.sin(self.pi2 * self.frequency * self.detune1 * relative_t)
        osc3: float = math.sin(self.pi2 * self.frequency * self.detune2 * relative_t)

        #
        return env * (osc1 + osc2 + osc3) / 3 * 0.2

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32]) -> NDArray[np.float32]:

        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer)
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
        ### Very slow attack and release (ASR envelope) (vectorized). ###
        #
        attack_val: NDArray[np.float32] = relative_t / self.attack
        sustain_val: NDArray[np.float32] = np.ones_like(relative_t)
        release_progress: NDArray[np.float32] = (relative_t - self.release_start) / self.release
        release_val: NDArray[np.float32] = 1.0 - release_progress

        #
        env: NDArray[np.float32] = np.where(
            relative_t < self.attack,
            attack_val,
            np.where(
                relative_t < self.release_start,
                sustain_val,
                release_val
            )
        )

        #
        ### Multiple detuned oscillators for chorus effect (vectorized). ###
        #
        phase_base: NDArray[np.float32] = self.pi2 * self.frequency * relative_t

        #
        osc1: NDArray[np.float32] = np.sin(phase_base)
        osc2: NDArray[np.float32] = np.sin(phase_base * self.detune1)
        osc3: NDArray[np.float32] = np.sin(phase_base * self.detune2)

        #
        ### Combine, average, apply gain and mask. ###
        #
        return (env * (osc1 + osc2 + osc3) / 3.0 * 0.2 * mask).astype(dtype=np.float32)
