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
from typing import List, Iterator
from dataclasses import dataclass

#
from nasong.theory.core.time import Duration, QUARTER, SIXTEENTH


@dataclass
class RhythmEvent:
    duration: Duration
    is_rest: bool = False
    velocity_scale: float = 1.0


@dataclass
class Rhythm:
    """
    A sequence of RhythmEvents.
    """

    events: List[RhythmEvent]
    loop: bool = True

    def __post_init__(self):
        # Validation or normalization
        pass

    def __iter__(self) -> Iterator[RhythmEvent]:
        if self.loop:
            import itertools

            return itertools.cycle(self.events)
        return iter(self.events)

    @classmethod
    def from_string(cls, pattern: str, unit: Duration = SIXTEENTH) -> "Rhythm":
        """
        Parses a string pattern.
        'x' or 'X' = Note
        '.' or '-' = Rest
        Example: "x..x" -> Note, Rest, Rest, Note (all 16th notes)
        """
        events = []
        for char in pattern:
            if char.lower() == "x":
                events.append(RhythmEvent(unit, is_rest=False))
            elif char in [".", "-"]:
                events.append(RhythmEvent(unit, is_rest=True))
            else:
                # ignore spaces or other chars?
                pass
        return cls(events)


# Common Factories
def four_on_the_floor() -> Rhythm:
    return Rhythm(
        [
            RhythmEvent(QUARTER, is_rest=False),
            RhythmEvent(QUARTER, is_rest=False),
            RhythmEvent(QUARTER, is_rest=False),
            RhythmEvent(QUARTER, is_rest=False),
        ]
    )


def swing_eighths() -> Rhythm:
    #
    long_d = Duration(1.0 * (2 / 3))
    short_d = Duration(1.0 * (1 / 3))

    return Rhythm(
        [
            RhythmEvent(long_d),
            RhythmEvent(short_d),
            RhythmEvent(long_d),
            RhythmEvent(short_d),
        ]
    )
