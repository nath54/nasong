import pytest
from nasong.theory.core.time import TimeSignature, Duration, WHOLE, QUARTER, dotted
from nasong.theory.systems.western import Western
from nasong.theory.core.pitch import Note

# ==========================================
# Time Tests
# ==========================================


def test_time_signature():
    ts = TimeSignature(4, 4)
    assert ts.beats_per_bar == 4
    assert ts.beat_value == 4
    assert ts.bar_length_in_quarters() == 4.0

    ts_68 = TimeSignature(6, 8)
    # 6 beats, each is an 8th note (0.5 quarters) -> 3.0 quarters
    assert ts_68.bar_length_in_quarters() == 3.0


def test_duration_math():
    d = QUARTER + QUARTER
    assert d.value == 2.0
    # WHOLE is 4.0, 4.0 * 0.5 = 2.0
    assert (WHOLE * 0.5).value == 2.0


def test_dotted():
    d = dotted(QUARTER)
    assert d.value == 1.5


# ==========================================
# Western System Tests
# ==========================================


def test_western_major():
    c_major = Western.major("C4")
    notes = c_major.notes
    assert len(notes) == 7
    # C, D, E, F, G, A, B
    assert notes[0].name == "C4"
    assert notes[2].name == "E4"  # Major 3rd
    assert notes[6].name == "B4"  # Major 7th


def test_western_minor():
    a_minor = Western.minor("A4")
    notes = a_minor.notes
    # A, B, C, D, E, F, G (Natural Minor)
    assert notes[0].name == "A4"
    assert notes[2].name == "C5"  # Minor 3rd up from A4


def test_western_modes():
    # Dorian: 1 2 b3 4 5 6 b7
    d_dorian = Western.dorian("D4")
    # D, E, F, G, A, B, C (all white keys)
    notes = d_dorian.notes
    assert notes[1].name == "E4"
    assert notes[2].name == "F4"  # Minor 3rd
    assert notes[5].name == "B4"  # Major 6th
    assert notes[6].name == "C5"  # Minor 7th


def test_western_phrygian():
    e_phrygian = Western.phrygian("E4")
    # E, F, G, A, B, C, D
    notes = e_phrygian.notes
    assert notes[1].name == "F4"  # Minor 2nd
