import pytest
from nasong.dsl.units import BPM, Ms, Bars
from nasong.dsl.utils import arpeggiate
from nasong.dsl.chain import Chainable, Gain, Processor
from nasong.theory.structures.chord import Chord
from nasong.theory.core.time import QUARTER
from nasong.core.values.basic.value_constant import Constant


# Units
def test_bpm_to_ms():
    b = BPM(120)
    # 60000 / 120 = 500ms per quarter
    assert b.to_ms(1.0) == 500.0
    assert b.to_ms(0.5) == 250.0


def test_ms_to_seconds():
    m = Ms(1500)
    assert m.to_seconds() == 1.5


# Utils
def test_arpeggiate():
    c_maj = Chord.from_name("C4", "major")
    # C4, E4, G4
    # Pattern [0, 1, 2, 0]
    # arpeggiate returns List[Note]
    notes = arpeggiate(c_maj, pattern=[0, 1, 2, 0], duration=QUARTER)
    assert len(notes) == 4
    assert notes[0].name == "C4"
    assert notes[1].name == "E4"
    assert notes[2].name == "G4"
    assert notes[3].name == "C4"


def test_arpeggiate_octaves():
    c_maj = Chord.from_name("C4", "major")
    # Pattern [0, 3] -> 0 is C4, 3 is 0th index + 1 octave (3 // 3 = 1) if pattern logic holds
    # My implementation: index // len(pitches) is octave shift.
    # 3 // 3 = 1. index % 3 = 0 -> C4 transposed up 1 octave -> C5
    notes = arpeggiate(c_maj, pattern=[0, 3], duration=QUARTER)
    assert notes[1].name == "C5"


# Chain
# We need a mock Value or use Constant
def test_chainable():
    from nasong.dsl.chain import Chainable, Gain

    c = Constant(1.0)
    # Wrapper
    # In DSL user code, we might want: Osc(440) >> Gain(0.5)
    # But Osc returns Value. We need it to return Chainable?
    # Or we monkey patch Value.

    # Testing explicit usage first
    ch = Chainable(c)

    # Gain is a Processor
    g = Gain(0.5)

    res = ch >> g
    assert isinstance(res, Chainable)
    # Result value should be Product(Constant(1.0), Constant(0.5))
    # We can check structure or evaluate?
    # Evaluate might need sample_rate etc.
    # Just check type for now
    from nasong.core.values.mult_itms_ops.value_product import Product

    assert isinstance(res.value, Product)
