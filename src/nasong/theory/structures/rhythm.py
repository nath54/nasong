"""
Rhythm Generators.
"""

from dataclasses import dataclass
from typing import List, Iterator, Tuple
from nasong.theory.core.time import Duration, QUARTER, SIXTEENTH, EIGHTH


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
    # Approximate swing feel with triplets?
    # Quarter note encoded as triplet pattern: 2/3 + 1/3
    # Or just use relative durations if engine supports it.
    # For now, let's use exact durations based on triplet math.
    long_8 = Duration(
        EIGHTH.value * (2 / 1.5)
    )  # Wait, normal 8th is 0.5. Triplet 8th is 1/3.
    # Swing pair: Long (2/3 of beat), Short (1/3 of beat). Beat = Quarter (1.0).
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
