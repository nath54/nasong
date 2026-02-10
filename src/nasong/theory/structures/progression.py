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
from typing import List, Union, Optional
from dataclasses import dataclass

#
from nasong.theory.core.scale import Scale
from nasong.theory.core.pitch import Pitch
from nasong.theory.structures.chord import Chord
from nasong.theory.core.time import Duration, QUARTER


@dataclass
class Progression:
    """
    A sequence of chords.
    """

    chords: List[Union[Chord, "Pitch"]]
    #
    ### Content for analysis/generation ###
    #
    scale: Optional[Scale] = None

    def __post_init__(self):
        #
        ### Allow passing Pitch objects and wrap them into simple Chords ###
        #
        if self.chords and isinstance(self.chords[0], Pitch):
            adapted = []
            for p in self.chords:
                #
                ### Wrap Pitch into a simple Chord with no intervals ###
                #
                adapted.append(Chord(p, intervals=[]))
            self.chords = adapted

    @property
    def duration(self) -> Duration:
        total = Duration(0.0)
        for c in self.chords:
            total = total + c.duration
        return total

    def __iter__(self):
        return iter(self.chords)

    def __getitem__(self, index):
        return self.chords[index]

    @classmethod
    def from_roman_numerals(
        cls, scale: Scale, numerals: List[str], duration: Duration = QUARTER
    ) -> "Progression":
        """
        Generate a progression from a scale and roman numerals.
        e.g. scale=C Major, numerals=["I", "IV", "V", "I"]
        """
        chords = []
        for roam in numerals:
            #
            ### Parse roman numeral ###
            #
            degree, quality = cls._parse_roman(roam)

            #
            ### Get root note from scale degree ###
            #
            root_note = scale.degree(degree)

            #
            ### Determine chord quality if not explicit ###
            #
            final_quality = quality
            if not quality:
                #
                ### Default based on case ###
                #
                if roam[0].isupper():
                    final_quality = "major"
                else:
                    final_quality = "minor"

            #
            ### Build chord ###
            #
            root_name = root_note.name

            chords.append(Chord.from_name(root_name, final_quality, duration))

        return cls(chords, scale)

    @staticmethod
    def _parse_roman(token: str):
        """
        Parse "IV", "vii", "V7", etc.
        Returns (degree_index, quality_hint).
        Degree index is 1-based (1..7).
        """
        #
        ### Very basic parser ###
        ### 1. Extract numerals ###
        #
        token = token.strip()

        #
        ### Mapping ###
        #
        romans = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7}

        #
        ### Check start of string for roman numeral match ###
        ### sort by length desc to match 'iii' before 'i' ###
        #
        sorted_keys = sorted(romans.keys(), key=len, reverse=True)

        degree = 1

        lower_token = token.lower()

        matched_key = ""
        for key in sorted_keys:
            if lower_token.startswith(key):
                matched_key = key
                degree = romans[key]
                break

        if not matched_key:
            raise ValueError(f"Invalid roman numeral: {token}")

        #
        ### remainder is quality/extension ###
        #
        remainder = token[len(matched_key) :]

        #
        ### Infer basic quality from case of the matched part (in original token) ###
        #
        original_roman = token[: len(matched_key)]
        is_major = original_roman[0].isupper()  # I, IV, V

        #
        ### Simple mapping for now ###
        #
        if remainder == "7":
            return degree, "dom7" if is_major else "min7"  # V7 -> dom7, i7 -> min7?
        if remainder == "maj7":
            return degree, "maj7"
        if remainder == "dim":
            return degree, "dim"

        #
        ### if no remainder, use case ###
        #
        if is_major:
            return degree, "major"
        else:
            return degree, "minor"
