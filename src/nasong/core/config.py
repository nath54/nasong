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
TODO: add full docstring, explaining what the goal of this script is, and explaining for each class and each function what is it, how it works, and how to use it.
"""


#
class Config:
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
