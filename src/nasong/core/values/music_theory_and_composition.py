# Copyright (C) 2026 Nathan Cerisara <https://github.com/nath54/nasong>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""
Music theory and composition utilities.

This module provides functions for converting musical notation to frequencies,
generating chord frequencies, and creating simple melodic sequences.
"""

#
### Import Modules. ###
#
from typing import Callable
from nasong.core.value import Value
from nasong.core.values.single_itms_ops.value_sequencer import Sequencer


#
### SYSTEM: MUSIC THEORY & COMPOSITION ###
#


#
def midi_note_to_freq(note_number: int) -> float:
    """Converts a MIDI note number to its equivalent frequency in Hz.

    Uses Equal Temperament with A4 = 69 = 440 Hz.

    Args:
        note_number (int): The MIDI note number (e.g., 60 for Middle C).

    Returns:
        float: The frequency in Hertz.
    """

    return 440.0 * (2.0 ** ((note_number - 69) / 12.0))


#
def get_chord_frequencies(root_freq: float, quality: str = "major") -> list[float]:
    """Generates a list of frequencies for a chord based on a root frequency.

    Uses Equal Temperament ratios relative to the root frequency.

    Args:
        root_freq (float): The fundamental frequency of the chord.
        quality (str, optional): The chord quality (e.g., "major", "minor",
            "diminished", "augmented", "maj7", "min7", "dom7").
            Defaults to "major".

    Returns:
        list[float]: A list of frequencies in Hz representing the chord.
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
    instrument_factory: Callable[..., Value],
    notes: list[tuple[float, float]],
    start_time: float = 0.0,
    gap: float = 0.0,
) -> Value:
    """Creates a monophonic melody sequence from frequency/duration pairs.

    This is a high-level wrapper around the `Sequencer` class. It automatically
    calculates start times for each note based on the provided durations and gaps.

    Args:
        time (Value): The master time input for the sequencer.
        instrument_factory (Callable[..., Value]): A function that takes
            (time, frequency, start_time, duration) and returns a Value object.
        notes (list[tuple[float, float]]): A list of (frequency, duration) tuples.
        start_time (float, optional): The start time of the first note in seconds.
            Defaults to 0.0.
        gap (float, optional): The silence gap between notes in seconds.
            Defaults to 0.0.

    Returns:
        Value: A Sequencer object that sums all notes in the melody.
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
