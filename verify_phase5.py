from nasong.theory.systems.raga import Raga
from nasong.theory.systems.maqam import Maqam
from nasong.theory.systems.gamelan import Gamelan
from nasong.theory.generators.styles.jazz import Jazz
from nasong.theory.generators.styles.edm import EDM
from nasong.theory.generators.styles.lofi import Lofi


def test_systems():
    print("Testing Systems...")
    r = Raga.create("C4", "bilawal")
    print(f"Raga Bilawal: {[i.semitones for i in r.intervals]}")

    m = Maqam.create("D4", "bayati")
    print(f"Maqam Bayati: {[i.semitones for i in m.intervals]}")

    g = Gamelan.create("C4", "slendro")
    print(f"Gamelan Slendro: {[i.semitones for i in g.intervals]}")


def test_styles():
    print("\nTesting Styles...")
    # Jazz
    jazz_prog = Jazz.ii_V_I("C4")
    print(f"Jazz ii-V-I: {[c.name for c in jazz_prog.chords]}")

    # EDM
    edm_prog = EDM.epic_chords("F4")
    print(f"EDM Epic: {[c.root.name for c in edm_prog.chords]}")

    # Lofi
    lofi_prog = Lofi.chill_progression("Db4")
    print(f"Lofi Chill: {[c.root.name for c in lofi_prog.chords]}")


if __name__ == "__main__":
    test_systems()
    test_styles()
