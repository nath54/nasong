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
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.basic.value_constant import Constant
from nasong.core.values.complex.value_sin import Sin
from nasong.core.values.mult_itms_ops.value_product import Product
from nasong.core.values.mult_itms_ops.value_sum import Sum
from nasong.core.values.single_itms_ops.value_sequencer import Sequencer


#
### SYSTEM: MUSIC THEORY & COMPOSITION ###
#


#
def midi_note_to_freq(note_number: int) -> float:
    """Converts MIDI note number to frequency (A4 = 69 = 440Hz)."""

    #
    ### Return the frequency. ###
    #
    return 440.0 * (2.0 ** ((note_number - 69) / 12.0))


#
def get_chord_frequencies(root_freq: float, quality: str = "major") -> list[float]:
    """
    Returns a list of frequencies for a chord based on a root frequency.
    This is a simplified frequency-ratio based approach (Just Intonation-ish),
    or we could map back to note numbers for Equal Temperament.
    Let's use Equal Temperament ratios relative to the root.
    """

    #
    ### Semitone multipliers. ###
    #
    semitone: float = 2.0 ** (1.0 / 12.0)

    #
    intervals: list[int] = []

    #
    ### Chord intervals based on quality. ###
    #
    if quality == "major":
        #
        intervals = [0, 4, 7]  # Root, Major 3rd, Perfect 5th
    #
    elif quality == "minor":
        #
        intervals = [0, 3, 7]  # Root, Minor 3rd, Perfect 5th
    #
    elif quality == "diminished":
        #
        intervals = [0, 3, 6]
    #
    elif quality == "augmented":
        #
        intervals = [0, 4, 8]
    #
    elif quality == "maj7":
        #
        intervals = [0, 4, 7, 11]
    #
    elif quality == "min7":
        #
        intervals = [0, 3, 7, 10]
    #
    elif quality == "dom7":
        #
        intervals = [0, 4, 7, 10]
    #
    else:
        #
        intervals = [0, 4, 7]  # Default to major

    #
    ### Return the chord frequencies. ###
    #
    return [root_freq * (semitone**i) for i in intervals]


#
def SimpleMelody(
    time: Value,
    instrument_factory: callable,
    notes: list[tuple[float, float]],  # (frequency, duration)
    start_time: float = 0.0,
    gap: float = 0.0,
) -> Value:
    """
    A simplified sequencer for monophonic melodies.
    Automatically calculates start times based on note durations.

    Args:
        notes: List of (frequency, duration) tuples.
    """

    #
    note_data_list: list[tuple[float, float, float]] = []
    current_time: float = start_time

    #
    ### Build the note data list. ###
    #
    for freq, dur in notes:
        #
        ### Assuming factory signature is (time, freq, start, dur, [amp/other]) ###
        ### We'll pass freq, start, dur. The factory needs to handle these. ###
        ### If the factory needs more args, this helper might need to be more generic or the factory wrapped. ###
        #
        note_data_list.append((freq, current_time, dur))
        #
        current_time += dur + gap

    #
    ### Return the sequencer. ###
    #
    return Sequencer(
        time, instrument_factory=instrument_factory, note_data_list=note_data_list
    )
