#
### Import Modules. ###
#
import lib_value as lv
import math
#
import numpy as np
from numpy.typing import NDArray


#
class ADSR_Piano(lv.Value):

    """
    Envelope generator: Attack, Decay, Sustain, Release.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": **POOR**. This is not a "true" ADSR envelope.
        - **Reason:** It uses the `modulo (%)` operator on time. This creates a
            *looping* envelope (an LFO), not a one-shot envelope that
            triggers once per note. It will sound "bad" for most instruments
            as it will constantly re-trigger the attack.
        - **Improvement:** This class should not be used. Use `ADSR2`,
            which is a "truthful" one-shot envelope. I have implemented
            the `getitem_np` to match the user's "bad" looping logic.
        - Note: The `note_freq` parameter is unused.
    """

    #
    def __init__(
        self,
        time: lv.Value,
        note_freq: float,  # This parameter is unused in the original code
        attack: float = 0.05,
        decay: float = 0.1,
        sustain_level: float = 0.7,
        release: float = 0.3,
        note_duration: float = 1.0
    ) -> None:

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.note_freq: float = note_freq
        self.attack: float = attack
        self.decay: float = decay
        self.sustain_level: float = sustain_level
        self.release: float = release
        self.note_duration: float = note_duration
        self.total_cycle_time: float = note_duration + release

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        #
        t_mod: float = t % self.total_cycle_time

        #
        ### Attack phase. ###
        #
        if t_mod < self.attack:
            #
            return t_mod / self.attack
        #
        ### Decay phase. ###
        #
        elif t_mod < self.attack + self.decay:
            #
            progress: float = (t_mod - self.attack) / self.decay
            #
            return 1.0 - (1.0 - self.sustain_level) * progress
        #
        ### Sustain phase. ###
        #
        elif t_mod < self.note_duration:
            #
            return self.sustain_level
        #
        ### Release phase. ###
        #
        elif t_mod < self.total_cycle_time:
            #
            progress: float = (t_mod - self.note_duration) / self.release
            #
            return self.sustain_level * (1.0 - progress)
        #
        else:
            #
            return 0.0

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        ### Get the time buffer and apply the looping modulo operator. ###
        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        #
        t_mod: NDArray[np.float32] = np.mod(t, self.total_cycle_time)

        #
        ### Define the 4 stages and their values. ###
        #
        attack_mask: NDArray[np.bool_] = t_mod < self.attack
        attack_val: NDArray[np.float32] = t_mod / self.attack

        #
        decay_mask: NDArray[np.bool_] = t_mod < (self.attack + self.decay)
        decay_progress: NDArray[np.float32] = (t_mod - self.attack) / self.decay
        decay_val: NDArray[np.float32] = (1.0 - (1.0 - self.sustain_level) * decay_progress).astype(dtype=np.float32)

        #
        sustain_mask: NDArray[np.bool_] = t_mod < self.note_duration
        sustain_val: NDArray[np.float32] = np.full_like(t_mod, self.sustain_level)

        #
        ### The final 'else' is the release phase. ###
        #
        release_progress: NDArray[np.float32] = ((t_mod - self.note_duration) / self.release).astype(dtype=np.float32)
        release_val: NDArray[np.float32] = (self.sustain_level * (1.0 - release_progress)).astype(dtype=np.float32)

        #
        ### Build the envelope with nested np.where ###
        ### np.where(condition, if_true, if_false)  ###
        #
        return np.where(
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
class ADSR2(lv.Value):

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
        time: lv.Value,
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
        self.time: lv.Value = time
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
class Distortion(lv.Value):

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
    def __init__(self, value: lv.Value, drive: float = 5.0) -> None:

        """
        Initializes the distortion effect.

        Args:
            value: The input `lv.Value` (the audio signal) to be distorted.
            drive: The amount of gain to apply before clipping.
                    Higher values = more distortion.
        """

        #
        super().__init__()

        #
        self.value: lv.Value = value
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


#
class Vibrato(lv.Value):

    """
    Generates a *frequency value* modulated by an LFO (vibrato).

        "Truthness" / "Good Listening" Analysis:
        - "Truthness": **EXCELLENT**. This is a "truthful" model of
            Frequency Modulation (FM), which is how vibrato is modeled
            in synthesis.
        - **Critical Note:** This is **NOT an audio effect**. It does not
            take an audio signal as input. It is a **Value Generator**,
            like `lv.Sin` or `lv.Constant`. It is intended to be used
            as the `frequency` input for an oscillator (like `lv.Sin`).
        - "Good Listening": N/A, as it doesn't produce sound directly.
            It is "good" for its intended purpose.
    """

    #
    def __init__(
        self,
        time: lv.Value,
        base_frequency: float,
        vibrato_rate: float = 5.0,  # The LFO frequency in Hz
        vibrato_depth: float = 0.015 # The modulation amount (e.g., 1.5%)
    ) -> None:

        """
        Initializes the vibrato frequency generator.

        Args:
            time: The global time `lv.Value`.
            base_frequency: The center frequency (in Hz).
            vibrato_rate: The speed of the wobble (LFO freq in Hz).
            vibrato_depth: The "width" of the wobble (as a percentage).
        """

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.base_frequency: float = base_frequency
        self.vibrato_rate: float = vibrato_rate
        self.vibrato_depth: float = vibrato_depth
        self.pi2: float = 2 * math.pi

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.__getitem__(index=index, sample_rate=sample_rate)

        #
        ### Calculate the LFO value (a sine wave oscillating between -1 and 1). ###
        #
        modulation: float = math.sin(self.pi2 * self.vibrato_rate * t)

        #
        ### Apply modulation to the base frequency. ###
        #
        return self.base_frequency * (1.0 + self.vibrato_depth * modulation)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        ### Get the time buffer. ###
        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)

        #
        ### Calculate the LFO value (a sine wave oscillating between -1 and 1). ###
        #
        modulation: NDArray[np.float32] = np.sin(self.pi2 * self.vibrato_rate * t)

        #
        ### Apply modulation to the base frequency. ###
        #
        return (self.base_frequency * (1.0 + self.vibrato_depth * modulation)).astype(dtype=np.float32)
