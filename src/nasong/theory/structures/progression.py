"""
Chord Progression Structure.
"""

from dataclasses import dataclass, field
from typing import List, Union, Optional
from nasong.theory.core.scale import Scale
from nasong.theory.structures.chord import Chord
from nasong.theory.core.time import Duration, QUARTER


@dataclass
class Progression:
    """
    A sequence of chords.
    """

    chords: List[Chord]
    scale: Optional[Scale] = None  # Content for analysis/generation

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
            # Parse roman numeral
            degree, quality = cls._parse_roman(roam)

            # Get root note from scale degree
            root_note = scale.degree(degree)

            # Determine chord quality if not explicit
            # Diatonic chords in Major:
            # I=Maj, ii=min, iii=min, IV=Maj, V=Maj, vi=min, vii=dim
            # If 'quality' is just from case (I vs i), use that.
            # If explicit (e.g. V7), use that.

            # Simple implementation:
            # if uppercase -> Major
            # if lowercase -> Minor
            # if 'dim' or 'o' -> Diminished
            # if '+' -> Augmented

            # Better approach: Use scale notes to build tertian chords?
            # Or just use the quality implied by the numeral string?

            final_quality = quality
            if not quality:
                # Default based on case
                if roam[0].isupper():
                    final_quality = "major"
                else:
                    final_quality = "minor"

            # Build chord
            # Note: root_note is a Pitch/Note object. Chord.from_name expects string root usually?
            # But we can instantiate Chord directly if we knew intervals.
            # Chord.from_name takes root NAME.
            # Let's use root name if it is a Note.
            root_name = root_note.name  # Logic assumes Note object

            chords.append(Chord.from_name(root_name, final_quality, duration))

        return cls(chords, scale)

    @staticmethod
    def _parse_roman(token: str):
        """
        Parse "IV", "vii", "V7", etc.
        Returns (degree_index, quality_hint).
        Degree index is 1-based (1..7).
        """
        # Very basic parser
        # 1. Extract numerals
        token = token.strip()

        # Mapping
        romans = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7}

        # Check start of string for roman numeral match
        # sort by length desc to match 'iii' before 'i'
        sorted_keys = sorted(romans.keys(), key=len, reverse=True)

        degree = 1
        quality = ""

        lower_token = token.lower()

        matched_key = ""
        for key in sorted_keys:
            if lower_token.startswith(key):
                matched_key = key
                degree = romans[key]
                break

        if not matched_key:
            raise ValueError(f"Invalid roman numeral: {token}")

        # remainder is quality/extension
        remainder = token[len(matched_key) :]

        # Infer basic quality from case of the matched part (in original token)
        original_roman = token[: len(matched_key)]
        is_major = original_roman[0].isupper()  # I, IV, V

        # Combine with remainder
        # e.g. "V7" -> Major + "7". -> "dom7"?
        # "vii" -> Minor + "" -> "minor"? But vii in major is dim.
        # This function just returns hints.
        # Ideally we specify exact quality mapping or let caller decide.

        # Simple mapping for now:
        if remainder == "7":
            return degree, "dom7" if is_major else "min7"  # V7 -> dom7, i7 -> min7?
        if remainder == "maj7":
            return degree, "maj7"
        if remainder == "dim":
            return degree, "dim"

        # if no remainder, use case
        if is_major:
            return degree, "major"
        else:
            return degree, "minor"
