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


"""Afrobeat style generators.

This module provides algorithmic generators for Afrobeat-style patterns,
leveraging West African pentatonic scales and polyrhythmic structures.
"""

#
### Import Modules. ###
#
from nasong.theory.core.time import QUARTER, SIXTEENTH
from nasong.theory.systems.african import African
from nasong.theory.structures.progression import Progression
from nasong.theory.structures.rhythm import Rhythm, RhythmEvent


class Afrobeat:
    """Generates Afrobeat style patterns and grooves."""

    @staticmethod
    def polyrhythmic_groove(root: str = "C4") -> Progression:
        """Generates a basic Afrobeat groove with a 3:2 polyrhythm structure.

        Builds a harmonic I-IV vamp and converts the polyrhythm onset data
        into ``Rhythm`` objects, which are attached to the returned
        ``Progression`` as ``rhythm_a`` and ``rhythm_b`` attributes.

        Args:
            root (str): The root note for the scale. Defaults to "C4".

        Returns:
            Progression: A rhythmic progression with ``rhythm_a`` (3-beat line)
                and ``rhythm_b`` (2-beat line) attached as attributes.
        """
        scale = African.pentatonic(root)
        # 3 against 2
        r_a, r_b = African.polyrhythm((3, 2), length=12)

        # Build Rhythm objects from onset indices
        def _onsets_to_rhythm(onsets: list[int], length: int) -> Rhythm:
            """Converts a list of onset step indices into a ``Rhythm``."""
            events: list[RhythmEvent] = []
            for step in range(length):
                is_hit = step in onsets
                events.append(RhythmEvent(SIXTEENTH, is_rest=not is_hit))
            return Rhythm(events, loop=True)

        rhythm_a = _onsets_to_rhythm(r_a, 12)
        rhythm_b = _onsets_to_rhythm(r_b, 12)

        # Harmonic foundation: simple I-IV vamp
        prog = Progression.from_roman_numerals(scale, ["I", "IV"], duration=QUARTER * 4)

        # Attach polyrhythm lines for consumers that need them
        prog.rhythm_a = rhythm_a  # type: ignore[attr-defined]
        prog.rhythm_b = rhythm_b  # type: ignore[attr-defined]

        return prog
