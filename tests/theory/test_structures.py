import pytest
from nasong.theory.structures.note import Note
from nasong.theory.structures.chord import Chord
from nasong.theory.structures.progression import Progression
from nasong.theory.structures.rhythm import Rhythm, RhythmEvent, four_on_the_floor
from nasong.theory.core.pitch import Note as CoreNote, Pitch
from nasong.theory.core.scale import Scale
from nasong.theory.core.time import QUARTER, EIGHTH
from nasong.theory import expand, track_from_progression

# ==========================================
# Note Tests
# ==========================================


def test_note_structure():
    n = Note("C4", duration=QUARTER, velocity=0.8)
    assert n.name == "C4"
    assert n.duration == QUARTER
    assert n.velocity == 0.8


def test_note_transpose():
    n = Note("C4")
    n2 = n.transpose(2)
    assert n2.name == "D4"


# ==========================================
# Chord Tests
# ==========================================


def test_chord_creation():
    # C Major Triad
    c_maj = Chord.from_name("C4", "major")
    pitches = c_maj.pitches
    assert len(pitches) == 3
    assert pitches[0].name == "C4"
    assert pitches[1].name == "E4"
    assert pitches[2].name == "G4"


def test_chord_inversion():
    c_maj_inv1 = Chord.from_name("C4", "major")
    c_maj_inv1.inversion = 1
    pitches = c_maj_inv1.pitches
    # E4, G4, C5
    assert pitches[0].name == "E4"
    assert pitches[2].midi == 72  # C5


# ==========================================
# Progression Tests
# ==========================================


def test_progression_creation():
    c_scale = Scale.from_name("C4", "major")
    prog = Progression.from_roman_numerals(c_scale, ["I", "IV", "V", "I"])
    assert len(prog.chords) == 4
    assert prog.chords[0].name == "major"  # I
    assert prog.chords[0].root.name == "C4"

    assert prog.chords[1].root.name == "F4"  # IV
    assert prog.chords[2].root.name == "G4"  # V


def test_track_from_progression():
    c_scale = Scale.from_name("C4", "major")
    # 2 chords, each 1.0 duration (default QUARTER)
    prog = Progression.from_roman_numerals(c_scale, ["I", "V"], duration=QUARTER)

    track = track_from_progression(prog)
    # 3 notes per chord * 2 chords = 6 events
    assert len(track) == 6

    # First chord notes start at 0.0
    assert track[0][0] == 0.0
    assert track[1][0] == 0.0
    assert track[2][0] == 0.0

    # Second chord notes start at 1.0 (after 1 quarter)
    assert track[3][0] == 1.0


# ==========================================
# Rhythm Tests
# ==========================================


def test_rhythm_parsing():
    r = Rhythm.from_string("x.x.")
    events = r.events
    assert len(events) == 4
    assert not events[0].is_rest
    assert events[1].is_rest


def test_rhythm_factory():
    r = four_on_the_floor()
    assert len(r.events) == 4
    assert all(not e.is_rest for e in r.events)
    assert r.events[0].duration == QUARTER
