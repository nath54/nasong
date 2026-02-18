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


"""Rhythmic pattern representations.

This module defines classes for `RhythmEvent` and `Rhythm`, allowing for the
creation and manipulation of rhythmic sequences. It supports string-based
parsing for quick pattern drafting.
"""

#
### Import Modules. ###
#
from typing import Iterator
from dataclasses import dataclass

import itertools

#
from nasong.theory.core.time import Duration, QUARTER, SIXTEENTH


@dataclass
class RhythmEvent:
    """A single point in a rhythmic sequence.

    Attributes:
        duration (Duration): The time length of the event.
        is_rest (bool): Whether the event is a rest or an active note.
        velocity_scale (float): Multiplier for the note's velocity.
    """

    duration: Duration
    is_rest: bool = False
    velocity_scale: float = 1.0


@dataclass
class Rhythm:
    """A collection of rhythmic events that can be looped or played once.

    Attributes:
        events (list[RhythmEvent]): The ordered sequence of events.
        loop (bool): Whether the pattern should repeat indefinitely.
    """

    events: list[RhythmEvent]
    loop: bool = True

    def __post_init__(self):
        # Validation or normalization
        pass

    def __iter__(self) -> Iterator[RhythmEvent]:
        if self.loop:
            return itertools.cycle(self.events)
        return iter(self.events)

    @classmethod
    def from_string(cls, pattern: str, unit: Duration = SIXTEENTH) -> "Rhythm":
        """Parses a string pattern into a Rhythm object.

        Notation:
            'x' or 'X': Active note pulse.
            '.' or '-': Rest pulse.

        Example:
            "x..x" -> [Note, Rest, Rest, Note] each of `unit` duration.

        Args:
            pattern (str): The string pattern to parse.
            unit (Duration, optional): The duration of each character pulse.
                Defaults to SIXTEENTH.

        Returns:
            Rhythm: The resulting rhythmic sequence.
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
    """Generates a standard 4/4 'four on the floor' kick pattern."""
    return Rhythm(
        [
            RhythmEvent(QUARTER, is_rest=False),
            RhythmEvent(QUARTER, is_rest=False),
            RhythmEvent(QUARTER, is_rest=False),
            RhythmEvent(QUARTER, is_rest=False),
        ]
    )


def swing_eighths() -> Rhythm:
    """Generates a triplet-based swing eighth note pattern."""
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
