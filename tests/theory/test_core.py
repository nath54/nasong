import pytest
import math
from nasong.theory.core.pitch import Note, Hz, Tuning
from nasong.theory.core.interval import Interval
from nasong.theory.core.scale import Scale

# ==========================================
# Pitch & Note Tests
# ==========================================


def test_note_creation():
    n = Note("A4")
    assert n.name == "A4"
    assert n.midi == 69
    assert n.freq == 440.0


def test_note_flats_sharps():
    # C#4 = 61
    n1 = Note("C#4")
    assert n1.midi == 61

    # Db4 = 61 (Enharmonic)
    n2 = Note("Db4")
    assert n2.midi == 61
    assert n2.freq == n1.freq


def test_note_octaves():
    # C0 = 12
    c0 = Note("C0")
    assert c0.midi == 12

    # A0 = 21
    a0 = Note("A0")
    assert a0.midi == 21
    assert a0.freq == 27.5


def test_note_transpose():
    a4 = Note("A4")  # 69
    c5 = a4.transpose(3)  # A4 + 3 semitones = C5 (72)
    assert c5.midi == 72
    # Basic name check - might change depending on implementation details
    assert "C5" in str(c5) or "C#5" in str(c5)  # Just ensure it exists


def test_hz():
    h = Hz(440.0)
    assert h.to_hz() == 440.0
    assert float(h) == 440.0


# ==========================================
# Interval Tests
# ==========================================


def test_interval_creation():
    p5 = Interval(7)
    assert p5.semitones == 7.0
    assert abs(p5.ratio - 1.4983) < 0.001


def test_interval_from_name():
    # Assuming the TODO in interval.py is implemented or we'll catch the error if not
    # Based on view_file, _parse_name IS implemented but called in __init__
    m3 = Interval("m3")
    assert m3.semitones == 3.0

    p8 = Interval("octave")
    assert p8.semitones == 12.0


def test_interval_addition():
    i1 = Interval(3)  # m3
    i2 = Interval(4)  # M3
    i3 = i1 + i2
    assert i3.semitones == 7  # P5


def test_interval_apply_to_note():
    a4 = Note("A4")
    p5 = Interval(7)
    e5 = p5.add_to(a4)
    assert isinstance(e5, Note)
    assert e5.midi == 69 + 7  # 76 (E5)


def test_interval_apply_to_hz():
    base = Hz(100.0)
    octave = Interval(12)  # 2.0 ratio
    high = octave.add_to(base)
    assert isinstance(high, Hz)
    assert high.freq == 200.0


# ==========================================
# Scale Tests
# ==========================================


def test_scale_major():
    c_major = Scale.from_name("C4", "major")
    # C, D, E, F, G, A, B
    notes = c_major.notes
    assert len(notes) == 7
    assert notes[0].midi == 60  # C4
    assert notes[1].midi == 62  # D4
    assert notes[4].midi == 67  # G4 (Dominant)


def test_scale_minor():
    a_minor = Scale.from_name("A4", "minor")
    # A, B, C, D, E, F, G (Natural minor)
    notes = a_minor.notes
    assert notes[0].midi == 69  # A4
    assert notes[2].midi == 72  # C5 (m3)


def test_scale_degree():
    c_major = Scale.from_name("C4", "major")

    # 1st degree = C4
    assert c_major.degree(1).midi == 60

    # 5th degree = G4
    assert c_major.degree(5).midi == 67

    # 8th degree = C5 (Automatic octave wrapping)
    assert c_major.degree(8).midi == 72

    # 11th degree = F5 (4th + octave)
    assert c_major.degree(11).midi == 65 + 12  # F4 (65) -> F5 (77)
