"""
DSL Unit Definitions.
"""

from dataclasses import dataclass
from nasong.theory.core.time import Duration


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
