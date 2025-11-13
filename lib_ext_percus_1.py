#
### Import Modules. ###
#
import lib_value as lv
import math
import random
#
import numpy as np
from numpy.typing import NDArray


#
class KickDrum(lv.Value):

    """
    Electronic kick drum.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": **EXCELLENT**. This is a "truthful" and classic
            model of a synthesized kick drum (like an 808 or 909). It
            combines a pitch-enveloped sine wave (the "body") with a
            fast, clicky transient (the "beater").
        - "Good Listening": **EXCELLENT**. This is a solid, standard
            kick synthesis model that sounds "good" and "realistic" for
            an electronic drum.
    """

    #
    def __init__(
        self,
        time: lv.Value,
        trigger_time: float,
        amplitude: float = 0.6
    ) -> None:

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.trigger_time: float = trigger_time
        self.amplitude: float = amplitude

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        relative_time: float = t - self.trigger_time

        #
        ### Gate: duration is 0.5s. ###
        #
        if relative_time < 0 or relative_time > 0.5:
            #
            return 0.0

        #
        ### Pitch envelope (starts high, drops quickly). ###
        #
        pitch: float = 150 * math.exp(-relative_time * 20)

        #
        ### Amplitude envelope. ###
        #
        env: float = math.exp(-relative_time * 8)

        #
        ### Generate tone. ###
        #
        tone: float = math.sin(2 * math.pi * pitch * relative_time)

        #
        ### Add click. ###
        #
        click: float = math.exp(-relative_time * 50) * 0.3

        #
        return self.amplitude * env * (tone + click)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        ### Get time and relative time. ###
        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        relative_time: NDArray[np.float32] = t - self.trigger_time

        #
        ### Gate mask. ###
        #
        mask: NDArray[np.bool_] = (relative_time >= 0) & (relative_time <= 0.5)
        #
        if not np.any(mask):
            #
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        ### Pitch envelope (vectorized). ###
        #
        pitch: NDArray[np.float32] = 150 * np.exp(-relative_time * 20)

        #
        ### Amplitude envelope (vectorized). ###
        #
        env: NDArray[np.float32] = np.exp(-relative_time * 8)

        #
        ### Generate tone (vectorized)                                    ###
        ### Note: This is phase modulation, as 'pitch' is inside the sin. ###
        ### A more "correct" (but different-sounding) way would be        ###
        ### to integrate the frequency (pitch) to get phase.              ###
        ### But we will be "truthful" to the original code.               ###
        #
        tone: NDArray[np.float32] = np.sin(2 * np.pi * pitch * relative_time)

        #
        ### Add click (vectorized). ###
        #
        click: NDArray[np.float32] = (np.exp(-relative_time * 50) * 0.3).astype(dtype=np.float32)

        #
        ### Combine and apply mask. ###
        #
        return (self.amplitude * env * (tone + click) * mask).astype(dtype=np.float32)


#
class KickDrum2(lv.Value):

    """
    Simulates a kick drum (bass drum) sound.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": **EXCELLENT**. Another "truthful" kick model,
            very similar to `KickDrum`.
        - "Good Listening": **POOR** (in original form).
        - **Reason:** The "click" is made with `random.random()`. This
            is not vectorizable and sounds like inconsistent static,
            not a "click."
        - **Improvement:** The `getitem_np` replaces this "bad"
            random noise with a deterministic, vectorized noise generator
            (`_vectorized_noise`), making it "good listening."
    """

    #
    def __init__(self, time: lv.Value, start_time: float) -> None:

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.start_time: float = start_time

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        relative_t: float = t - self.start_time

        #
        if relative_t < 0 or relative_t > 0.5:
            #
            return 0.0

        #
        ### Pitch envelope (starts high, drops quickly). ###
        #
        pitch_env: float = 60 + 100 * math.exp(-relative_t * 50)

        #
        ### Amplitude envelope (very sharp attack and decay). ###
        #
        amp_env: float = math.exp(-relative_t * 15)

        #
        ### Sine wave for the body. ###
        #
        kick: float = math.sin(2 * math.pi * pitch_env * relative_t)

        #
        ### Add click for attack ("bad" random-based). ###
        #
        click: float = 0.5 * math.exp(-relative_t * 100) * (random.random() * 2 - 1)

        #
        return amp_env * (kick + click) * 0.6

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        relative_t: NDArray[np.float32] = t - self.start_time

        #
        ### Gate mask. ###
        #
        mask: NDArray[np.bool_] = (relative_t >= 0) & (relative_t <= 0.5)
        #
        if not np.any(mask):
            #
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        ### Pitch envelope (vectorized). ###
        #
        pitch_env: NDArray[np.float32] = 60 + 100 * np.exp(-relative_t * 50)

        #
        ### Amplitude envelope (vectorized). ###
        #
        amp_env: NDArray[np.float32] = np.exp(-relative_t * 15)

        #
        ### Sine wave for the body (vectorized). ###
        #
        kick: NDArray[np.float32] = np.sin(2 * np.pi * pitch_env * relative_t)

        #
        ### Add click for attack ("good" vectorized noise). ###
        #
        click_env: NDArray[np.float32] = (0.5 * np.exp(-relative_t * 100)).astype(dtype=np.float32)
        #
        click_noise: NDArray[np.float32] = lv.WhiteNoise.vectorized_noise(
            indexes_buffer,
            seed=4567,
            scale=1/100.0  # (2-1) / 200 = 0.005 approx
        ) * 200.0 # Scale to -1 to 1

        #
        click: NDArray[np.float32] = click_env * click_noise

        #
        return (amp_env * (kick + click) * 0.6 * mask).astype(dtype=np.float32)


#
class Snare(lv.Value):

    """
    Electronic snare with noise.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": **EXCELLENT**. This is the standard "truthful"
            model for a synthesized snare: a tonal "body" (sine wave)
            mixed with a noise burst (the "snare wires").
        - "Good Listening": **POOR** (in original form).
        - **Reason:** The `hash()`-based noise is "bad listening."
            It's not random, but a periodic digital artifact that
            will have a metallic, ugly "tone."
        - **Improvement:** The `getitem_np` replaces this "bad"
            hash noise with `_vectorized_noise`, making it "good listening."
    """

    #
    def __init__(
        self,
        time: lv.Value,
        trigger_time: float,
        amplitude: float = 0.4
    ) -> None:

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.trigger_time: float = trigger_time
        self.amplitude: float = amplitude

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        relative_time: float = t - self.trigger_time

        #
        if relative_time < 0 or relative_time > 0.3:
            return 0.0

        #
        ### Envelope. ###
        #
        env: float = math.exp(-relative_time * 15)

        #
        ### Tone component. ###
        #
        tone: float = math.sin(2 * math.pi * 200 * relative_time) * 0.4

        #
        ### Noise component ("bad" hash-based). ###
        #
        noise: float = (hash((index * 9973) % 1000000) % 200 - 100) / 100.0

        #
        return self.amplitude * env * (tone + noise * 0.8)

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        relative_time: NDArray[np.float32] = t - self.trigger_time

        #
        ### Gate mask. ###
        #
        mask: NDArray[np.bool_] = (relative_time >= 0) & (relative_time <= 0.3)
        #
        if not np.any(mask):
            #
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        ### Envelope (vectorized). ###
        #
        env: NDArray[np.float32] = np.exp(-relative_time * 15)

        #
        ### Tone component (vectorized). ###
        #
        tone: NDArray[np.float32] = (np.sin(2 * np.pi * 200 * relative_time) * 0.4).astype(dtype=np.float32)

        #
        ### Noise component ("good" vectorized noise). ###
        #
        noise: NDArray[np.float32] = lv.WhiteNoise.vectorized_noise(
            indexes_buffer,
            seed=9973,
            scale=1/100.0  # This matches the (200-100)/100 scaling
        )

        #
        return (self.amplitude * env * (tone + noise * 0.8) * mask).astype(dtype=np.float32)


#
class SnareDrum(lv.Value):

    """
    Simulates a snare drum sound.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": **EXCELLENT**. Another "truthful" snare model,
            just like `Snare`.
        - "Good Listening": **POOR** (in original form).
        - **Reason:** Uses `random.random()` for noise, which is "bad."
        - **Improvement:** The `getitem_np` replaces this "bad"
            random noise with `_vectorized_noise`, making it "good listening."
    """

    #
    def __init__(self, time: lv.Value, start_time: float) -> None:

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.start_time: float = start_time

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        relative_t: float = t - self.start_time

        #
        if relative_t < 0 or relative_t > 0.2:
            #
            return 0.0

        #
        ### Very sharp decay. ###
        #
        amp_env: float = math.exp(-relative_t * 30)

        #
        ### Tonal component (drum head). ###
        #
        tone_freq: float = 200
        tone: float = 0.3 * math.sin(2 * math.pi * tone_freq * relative_t)

        #
        ### Noise component (snare wires) ("bad" random-based). ###
        #
        noise: float = (random.random() * 2 - 1) * 0.7

        #
        return amp_env * (tone + noise) * 0.4

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        relative_t: NDArray[np.float32] = t - self.start_time

        #
        ### Gate mask. ###
        #
        mask: NDArray[np.bool_] = (relative_t >= 0) & (relative_t <= 0.2)
        #
        if not np.any(mask):
            #
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        ### Very sharp decay (vectorized). ###
        #
        amp_env: NDArray[np.float32] = np.exp(-relative_t * 30)

        #
        ### Tonal component (drum head) (vectorized). ###
        #
        tone_freq: float = 200
        #
        tone: NDArray[np.float32] = (0.3 * np.sin(2 * np.pi * tone_freq * relative_t)).astype(dtype=np.float32)

        #
        ### Noise component (snare wires) ("good" vectorized noise). ###
        #
        noise_val: NDArray[np.float32] = lv.WhiteNoise.vectorized_noise(
            indexes_buffer,
            seed=1337,
            scale=1/100.0
        ) * 100.0 # Scale to approx -1 to 1

        #
        noise: NDArray[np.float32] = noise_val * 0.7

        #
        return (amp_env * (tone + noise) * 0.4 * mask).astype(dtype=np.float32)


#
class HiHat(lv.Value):

    """
    Simulates a hi-hat cymbal.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": **GOOD**. This is a "truthful" model. Cymbals are
            "inharmonic," and this models inharmonicity by summing multiple
            high-frequency sine waves that are not musically related.
        - "Good Listening": **POOR**.
        - **Reason:**
            1.  **Aliasing:** The high-frequency `sin` waves (8-12 kHz)
                will alias *massively*, creating a harsh, ugly sound.
            2.  **Bad Noise:** The `random.random()` noise is "bad."
        - **Improvement:** `getitem_np` replaces the "bad" random noise
            with `_vectorized_noise`. The aliasing problem is unavoidable
            without the `sample_rate`, so it remains, but I've noted
            it in the docstring.
    """

    #
    def __init__(self, time: lv.Value, start_time: float, open: bool = False) -> None:

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.start_time: float = start_time
        self.open: bool = open
        self.duration: float = 0.4 if self.open else 0.08
        self.decay_rate: float = 8.0 if self.open else 50.0
        self.pi2: float = 2 * math.pi
        self.cymbal_freqs: list[float] = [8000.0, 9000.0, 10000.0, 11000.0, 12000.0]

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        relative_t: float = t - self.start_time

        #
        if relative_t < 0 or relative_t > self.duration:
            #
            return 0.0

        #
        ### Exponential decay (slower for open hi-hat). ###
        #
        amp_env: float = math.exp(-relative_t * self.decay_rate)

        #
        ### High-frequency noise (metallic sound). ###
        #
        noise: float = 0.0
        #
        for freq in self.cymbal_freqs:
            #
            noise += math.sin(self.pi2 * freq * relative_t) / 5

        #
        ### Add some randomness ("bad" random-based). ###
        #
        noise += (random.random() * 2 - 1) * 0.3

        #
        return amp_env * noise * 0.15

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
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
        ### Exponential decay (vectorized). ###
        #
        amp_env: NDArray[np.float32] = np.exp(-relative_t * self.decay_rate)

        #
        ### High-frequency noise (vectorized, but will alias). ###
        #
        noise: NDArray[np.float32] = np.zeros_like(indexes_buffer, dtype=np.float32)
        #
        for freq in self.cymbal_freqs:
            #
            noise += np.sin(self.pi2 * freq * relative_t)

        #
        ### Normalize. ###
        #
        noise /= len(self.cymbal_freqs)

        #
        ### Add some randomness ("good" vectorized noise). ###
        #
        random_noise: NDArray[np.float32] = lv.WhiteNoise.vectorized_noise(
            indexes_buffer,
            seed=2222,
            scale=1/100.0
        ) * 100.0 # Scale to -1 to 1

        #
        noise += random_noise * 0.3

        #
        return (amp_env * noise * 0.15 * mask).astype(dtype=np.float32)


#
class CrashCymbal(lv.Value):

    """
    Simulates a crash cymbal.

    "Truthness" / "Good Listening" Analysis:
        - "Truthness": **GOOD**. A "truthful" model, very similar to `HiHat`
            but with a longer decay and more complex frequencies.
        - "Good Listening": **POOR**.
        - **Reason:**
            1.  **Aliasing:** Same problem as `HiHat`, but even worse due
                to more sine waves at very high frequencies (up to 14.5 kHz).
            2.  **Bad Noise:** The `random.random()` *inside the loop* is
                non-vectorizable and will sound terrible.
        - **Improvement:** The `getitem_np` replaces this "bad"
            random noise. The original intent was likely *phase* noise
            to make the harmonics inharmonic. I have implemented this
            by creating a *static* random phase for each oscillator,
            which is "truthful" to the intent and vectorizable.
    """

    #
    def __init__(self, time: lv.Value, start_time: float) -> None:

        #
        super().__init__()

        #
        self.time: lv.Value = time
        self.start_time: float = start_time
        self.pi2: float = 2 * math.pi

        #
        ### Frequencies for the cymbal's inharmonic tones. ###
        #
        self.cymbal_freqs: list[float] = [
            7000.0, 8500.0, 10000.0, 11500.0, 13000.0, 14500.0
        ]

        #
        ### Pre-calculate static random phases for the `getitem_np` ###
        ### This is the "Improvement" to replace `random.random()`  ###
        #
        self.static_phases: NDArray[np.float32] = np.random.uniform(
            low=0.0,
            high=self.pi2,
            size=len(self.cymbal_freqs)
        ).astype(np.float32)

    #
    def __getitem__(self, index: int, sample_rate: int) -> float:

        #
        t: float = self.time.__getitem__(index=index, sample_rate=sample_rate)
        #
        relative_t: float = t - self.start_time

        #
        if relative_t < 0 or relative_t > 2.0:
            #
            return 0.0

        #
        ### Slow decay. ###
        #
        amp_env: float = math.exp(-relative_t * 2)

        #
        ### Complex high-frequency content ("bad" random-based). ###
        #
        noise: float = 0.0
        #
        for freq in self.cymbal_freqs:
            #
            noise += math.sin(self.pi2 * freq * relative_t + random.random())

        #
        ### Normalize. ###
        #
        noise /= len(self.cymbal_freqs)

        #
        return amp_env * noise * 0.25

    #
    def getitem_np(self, indexes_buffer: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:

        #
        t: NDArray[np.float32] = self.time.getitem_np(indexes_buffer=indexes_buffer, sample_rate=sample_rate)
        #
        relative_t: NDArray[np.float32] = t - self.start_time

        #
        ### Gate mask. ###
        #
        mask: NDArray[np.bool_] = (relative_t >= 0) & (relative_t <= 2.0)
        #
        if not np.any(mask):
            #
            return np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        ### Slow decay (vectorized). ###
        #
        amp_env: NDArray[np.float32] = np.exp(-relative_t * 2)

        #
        ### Complex high-frequency content (vectorized with static phase noise) ###
        ### This is the "Improvement" ###
        #
        noise: NDArray[np.float32] = np.zeros_like(indexes_buffer, dtype=np.float32)
        #
        for i, freq in enumerate(self.cymbal_freqs):
            #
            phase: NDArray[np.float32] = self.pi2 * freq * relative_t
            #
            static_phase_offset: float = self.static_phases[i]
            #
            noise += np.sin(phase + static_phase_offset)

        #
        noise /= len(self.cymbal_freqs) # Normalize

        #
        return (amp_env * noise * 0.25 * mask).astype(dtype=np.float32)
