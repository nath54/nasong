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
### Import Modules. ###
#
from dataclasses import dataclass


@dataclass
class BPM:
    value: float

    def to_ms(self, note_duration: float = 1.0) -> float:
        """
        Convert note duration (in quarters) to milliseconds at this BPM.
        60000 / BPM = ms per quarter note.
        """
        ms_per_quarter = 60000.0 / self.value
        return ms_per_quarter * note_duration


@dataclass
class Ms:
    value: float  # milliseconds

    def to_seconds(self) -> float:
        return self.value / 1000.0


@dataclass
class Bars:
    value: float  # in bars


@dataclass
class Hz:
    value: float  # frequency
