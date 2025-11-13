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
