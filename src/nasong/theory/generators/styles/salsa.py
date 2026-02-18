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


"""Salsa and Latin style generators.

This module provides generators for Salsa-style harmonic montunos and standard
Latin rhythmic patterns like the Son Clave.
"""

#
### Import Modules. ###
#
from nasong.theory.systems.western import Western
from nasong.theory.structures.rhythm import Rhythm
from nasong.theory.structures.progression import Progression


class Salsa:
    """Generates Salsa harmonic vamps (montunos) and rhythmic clave patterns."""

    @staticmethod
    def montuno_progression(root: str = "G4", minor: bool = True) -> Progression:
        """Generates a basic i-V salsa montuno progression."""
        scale = Western.minor(root) if minor else Western.major(root)
        return Progression.from_roman_numerals(scale, ["i", "V"])

    @staticmethod
    def clave_rhythm(direction: str = "2-3") -> Rhythm:
        """Returns a pulse pattern for Son Clave (2-3 or 3-2).

        The Son Clave is a foundational rhythmic pattern in Afro-Cuban music,
        expressed here as a 16-step (sixteenth-note) grid.

        Args:
            direction (str): Either ``"2-3"`` or ``"3-2"`` for the clave
                direction. Defaults to ``"2-3"``.

        Returns:
            Rhythm: A looping ``Rhythm`` of 16 sixteenth-note pulses.

        Raises:
            ValueError: If ``direction`` is not ``"2-3"`` or ``"3-2"``.
        """
        # 16-step grid (one bar of 4/4 in sixteenth notes)
        # 2-3 Son Clave: ..x.x.....x..x.x...  (side 2 then side 3)
        # 3-2 Son Clave: x..x..x...x.x.......  (side 3 then side 2)
        if direction == "2-3":
            return Rhythm.from_string("..x.x.....x..x.x...")
        if direction == "3-2":
            return Rhythm.from_string("x..x..x...x.x......")
        raise ValueError(f"Unknown clave direction: {direction!r} (use '2-3' or '3-2')")
