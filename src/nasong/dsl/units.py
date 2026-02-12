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


"""Musical and physical units for the NaSong DSL.

This module provides specialized dataclasses for handling common units like
BPM (Beats Per Minute), Ms (Milliseconds), Bars, and Hz (Hertz). It facilitates
clean unit conversions between musical time and physical time/frequency.
"""

#
### Import Modules. ###
#
from dataclasses import dataclass


@dataclass
class BPM:
    """Represents tempo in Beats Per Minute.

    Attributes:
        value (float): The tempo value.
    """

    value: float

    def to_ms(self, note_duration: float = 1.0) -> float:
        """Converts a note duration (in quarter notes) to milliseconds.

        Args:
            note_duration (float, optional): Duration in quarter notes (e.g., 1.0
                for a quarter note, 0.5 for an eighth). Defaults to 1.0.

        Returns:
            float: The duration in milliseconds.
        """
        ms_per_quarter = 60000.0 / self.value
        return ms_per_quarter * note_duration


@dataclass
class Ms:
    """Represents time in milliseconds.

    Attributes:
        value (float): The time value in milliseconds.
    """

    value: float  # milliseconds

    def to_seconds(self) -> float:
        """Converts milliseconds to seconds.

        Returns:
            float: The time value in seconds.
        """
        return self.value / 1000.0


@dataclass
class Bars:
    """Represents musical time in bars.

    Attributes:
        value (float): The number of bars.
    """

    value: float  # in bars


@dataclass
class Hz:
    """Represents frequency in Hertz.

    Attributes:
        value (float): The frequency value in Hz.
    """

    value: float  # frequency
