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


"""Plucked string instrument definitions.

This module provides factory functions for guitar and other plucked string
emulations. It includes both individual string synthesis and higher-level
"strumming" and "fingerpicking" sequencers.
"""

#
### Import Modules. ###
#
import math

#
import nasong.core.all_values as lv


#
### CATEGORY: PLUCKED STRINGS ###
#


def GuitarString(
    time: lv.Value,
    frequency: float,
    start_time: float,
    duration: float = 3.0,
    brightness: float = 1.0,
) -> lv.Value:
    """Simulates a single plucked guitar string.

    Built using additive synthesis with six harmonics and a dynamic brightness
    decay that simulates the natural loss of high-frequency energy over time.

    Args:
        time (lv.Value): The global time provider.
        frequency (float): The fundamental frequency of the string.
        start_time (float): Activation time in seconds.
        duration (float, optional): Length of the string vibration. Defaults to 3.0.
        brightness (float, optional): Initial harmonic richness. Defaults to 1.0.

    Returns:
        lv.Value: The audio value graph for the guitar string.
    """

    #
    ### Envelope: 0.002s attack, then exp(-t * 1.2) decay  ###
    #
    amp_env: lv.Value = lv.ExponentialADSR(
        time,
        note_start=start_time,
        note_duration=duration,
        attack_time=0.002,
        decay_time=duration - 0.002,
        sustain_level=0.0,
        release_time=0.01,
        attack_curve=1.0,  # Linear attack
        decay_curve=1.5,  # Approx exp(-t*1.2)
    )

    #
    ### Oscillator: 6 additive harmonics  ###
    #
    relative_time: lv.Value = lv.BasicScaling(time, lv.c(1), lv.c(-start_time))
    pi2: float = 2 * math.pi
    #
    h1: lv.Value = lv.Sin(relative_time, lv.c(frequency * 1 * pi2), lv.c(1.0))
    h2: lv.Value = lv.Sin(relative_time, lv.c(frequency * 2 * pi2), lv.c(0.6))
    h3: lv.Value = lv.Sin(relative_time, lv.c(frequency * 3 * pi2), lv.c(0.4))
    h4: lv.Value = lv.Sin(relative_time, lv.c(frequency * 4 * pi2), lv.c(0.25))
    h5: lv.Value = lv.Sin(relative_time, lv.c(frequency * 5 * pi2), lv.c(0.15))
    h6: lv.Value = lv.Sin(relative_time, lv.c(frequency * 6 * pi2), lv.c(0.1))

    #
    ### Brightness decay: exp(-t * 2 * brightness)  ###
    #
    brightness_env: lv.Value = lv.ExponentialDecay(time, start_time, 2.0 * brightness)

    #
    ### Apply brightness decay to harmonics per original logic  ###
    #
    df_1: lv.Value = lv.Sum(lv.c(0.5), lv.Product(lv.c(0.5), brightness_env))
    df_2: lv.Value = brightness_env
    df_3: lv.Value = lv.Product(brightness_env, lv.c(0.8))
    df_4: lv.Value = lv.Product(brightness_env, lv.c(0.6))
    df_5: lv.Value = lv.Product(brightness_env, lv.c(0.4))

    #
    signal: lv.Value = lv.Sum(
        h1,
        lv.Product(h2, df_1),
        lv.Product(h3, df_2),
        lv.Product(h4, df_3),
        lv.Product(h5, df_4),
        lv.Product(h6, df_5),
    )

    #
    ### Final = 0.25 * AmpEnv * Signal ###
    #
    return lv.Product(lv.c(0.25), amp_env, signal)


#
def GuitarString2(
    time: lv.Value,
    frequency: float,
    start_time: float,
    duration: float,
    amplitude: float = 0.4,
) -> lv.Value:
    """Creates an alternate guitar string sound with subtle noise.

    Args:
        time (lv.Value): Global time value.
        frequency (float): Fundamental frequency (Hz).
        start_time (float): Strike time in seconds.
        duration (float): Length of the note in seconds.
        amplitude (float, optional): Overall volume scaling. Defaults to 0.4.

    Returns:
        lv.Value: The audio value graph for the alternate guitar string.
    """

    #
    ### Envelope: Complex ADR . Approximated. ###
    #
    amp_env: lv.Value = lv.ExponentialADSR(
        time,
        note_start=start_time,
        note_duration=duration,
        attack_time=0.01,
        decay_time=duration - 0.11,  # 0.1s release
        sustain_level=0.0,  # Decays to 0
        release_time=0.1,
        attack_curve=1.0,
        decay_curve=1.2,  # Approx exp(-t*0.5)
    )

    #
    ### Oscillator: 3 harmonics, using `time` (t)  ###
    #
    pi2: float = 2 * math.pi
    #
    h1: lv.Value = lv.Sin(time, lv.c(frequency * 1 * pi2), lv.c(1.0))
    h2: lv.Value = lv.Sin(time, lv.c(frequency * 2 * pi2), lv.c(0.5))
    h3: lv.Value = lv.Sin(time, lv.c(frequency * 3 * pi2), lv.c(0.3))

    #
    ### Noise: Fixed with WhiteNoise  ###
    #
    noise: lv.Value = lv.WhiteNoise(seed=12345, scale=1 / 5000.0)

    #
    signal: lv.Value = lv.Sum(h1, h2, h3, noise)

    #
    ### Final = Amplitude * AmpEnv * Signal ###
    #
    return lv.Product(lv.c(amplitude), amp_env, signal)


#
def AcousticString(
    time: lv.Value,
    frequency: float,
    pluck_time: float,
    amplitude: float = 0.3,
    decay_rate: float = 2.0,
) -> lv.Value:
    """Simulates a bright acoustic string pluck.

    Args:
        time (lv.Value): Global time provider.
        frequency (float): Fundamental frequency (Hz).
        pluck_time (float): Pluck event time in seconds.
        amplitude (float, optional): Overall volume. Defaults to 0.3.
        decay_rate (float, optional): Speed of volume decay. Defaults to 2.0.

    Returns:
        lv.Value: The audio value graph for the acoustic string.
    """

    #
    ### Envelope: 0.005s attack  ###
    #
    attack_env: lv.Value = lv.ADSR2(time, pluck_time, 0.005, 0.005, 0.001, 1.0, 0.001)
    #
    decay_env: lv.Value = lv.ExponentialDecay(time, pluck_time, decay_rate)
    #
    amp_env: lv.Value = lv.Product(attack_env, decay_env)

    #
    ### Gate: 3.0s hard gate  ###
    #
    gate_env: lv.Value = lv.ADSR2(time, pluck_time, 3.0, 0.001, 0.001, 1.0, 0.001)

    #
    ### Oscillator: 5 harmonics, using `time` (t)  ###
    #
    pi2: float = 2 * math.pi
    #
    h1: lv.Value = lv.Sin(time, lv.c(frequency * 1 * pi2), lv.c(1.0))
    h2: lv.Value = lv.Sin(time, lv.c(frequency * 2 * pi2), lv.c(0.6))
    h3: lv.Value = lv.Sin(time, lv.c(frequency * 3 * pi2), lv.c(0.4))
    h4: lv.Value = lv.Sin(time, lv.c(frequency * 4 * pi2), lv.c(0.25))
    h5: lv.Value = lv.Sin(time, lv.c(frequency * 5 * pi2), lv.c(0.15))

    #
    ### Noise: Fixed with WhiteNoise  ###
    #
    noise: lv.Value = lv.WhiteNoise(seed=8191, scale=1 / 8000.0)

    #
    signal: lv.Value = lv.Sum(h1, h2, h3, h4, h5, noise)

    #
    ### Final = Amplitude * Gate * AmpEnv * Signal ###
    #
    return lv.Product(lv.c(amplitude), gate_env, amp_env, signal)


#
def Fingerpicking(
    time: lv.Value,
    bass_note: float,
    chord_notes: list[float],
    start_time: float,
    pattern_duration: float = 2.0,
) -> lv.Value:
    """Sequences an acoustic fingerpicking pattern.

    Creates a composite sound of alternating bass and treble notes over a fixed
    duration pattern.

    Args:
        time (lv.Value): Global time provider.
        bass_note (float): Frequency of the bass (root) note.
        chord_notes (list[float]): List of treble note frequencies.
        start_time (float): Start time of the entire pattern.
        pattern_duration (float, optional): Length of one full loop. Defaults to 2.0.

    Returns:
        lv.Value: A `Sequencer` object containing the fingerpicked pattern.
    """

    #
    note_data_list: list[tuple[float, ...]] = []

    #
    ### Bass notes  ###
    #
    note_data_list.append((bass_note, start_time, 0.35, 1.5))
    note_data_list.append((bass_note, start_time + pattern_duration / 2, 0.35, 1.5))

    #
    ### Treble notes  ###
    #
    eighth: float = pattern_duration / 8
    #
    for i, note_idx in enumerate([0, 1, 2, 1, 0, 1, 2, 1]):
        #
        if i % 2 == 1:  # Off-beats
            #
            pluck_time: float = start_time + i * eighth
            note: float = chord_notes[note_idx % len(chord_notes)]
            #
            note_data_list.append((note, pluck_time, 0.25, 2.0))

    #
    ### Factory function to create the notes ###
    #
    def acoustic_string_factory(
        t: lv.Value, freq: float, p_time: float, amp: float, d_rate: float
    ) -> lv.Value:
        #
        return AcousticString(t, freq, p_time, amp, d_rate)

    #
    return lv.Sequencer(
        time, instrument_factory=acoustic_string_factory, note_data_list=note_data_list
    )


#
def Strum(
    time: lv.Value, frequencies: list[float], start_time: float, duration: float = 2.5
) -> lv.Value:
    """Sequences a strummed chord across multiple frequencies.

    Each frequency is triggered with a slight time offset to simulate the
    physical movement of a pick or finger across strings.

    Args:
        time (lv.Value): Global time provider.
        frequencies (list[float]): List of string frequencies to strum.
        start_time (float): The start time of the strum.
        duration (float, optional): Gated duration for each string. Defaults to 2.5.

    Returns:
        lv.Value: A `Sequencer` object containing the strummed chord.
    """

    #
    note_data_list: list[tuple[float, ...]] = []

    #
    for i, freq in enumerate(frequencies):
        #
        ### Each string plucked slightly after  ###
        #
        offset: float = i * 0.015
        #
        ### Note Data: (frequency, start_time, duration, brightness) ###
        #
        note_data_list.append(
            (freq, start_time + offset, duration, 1.0)  # Default brightness
        )

    #
    ### Factory function to create the notes ###
    #
    def guitar_string_factory(
        t: lv.Value, freq: float, s_time: float, dur: float, bright: float
    ) -> lv.Value:
        #
        return GuitarString(t, freq, s_time, dur, bright)

    #
    return lv.Sequencer(
        time, instrument_factory=guitar_string_factory, note_data_list=note_data_list
    )
