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
Configuration management for audio generation.

This module provides the `Config` class, which holds global settings for the
audio engine, such as sample rate, duration, and output file naming.
"""


#
class Config:
    """Holds configuration settings for an audio generation session.

    Attributes:
        sample_rate (int): The number of samples per second (e.g., 44100).
        total_duration (float): The length of the generated audio in seconds.
        output_filename (str): The path or name for the saved WAV file.
    """

    #
    def __init__(
        #
        self,
        #
        ### The sample rate (samples per second) is a standard for CD quality audio. ###
        #
        sample_rate: int = 44100,
        #
        ### The duration of the sound in seconds. ###
        #
        total_duration: float = 40.0,
        #
        ### The output filename. ###
        #
        output_filename: str = "generated_sine_wave.wav",
        #
    ) -> None:
        """Initializes the Config object.

        Args:
            sample_rate (int, optional): Audio sample rate in Hz. Defaults to 44100.
            total_duration (float, optional): Duration in seconds. Defaults to 40.0.
            output_filename (str, optional): Target WAV filename.
                Defaults to "generated_sine_wave.wav".
        """

        #
        ### The sample rate (samples per second) is a standard for CD quality audio. ###
        #
        self.sample_rate: int = sample_rate
        #
        ### The duration of the sound in seconds. ###
        #
        self.total_duration: float = total_duration
        #
        ### The output filename. ###
        #
        self.output_filename: str = output_filename
