from nasong.theory.systems.african import African
from nasong.theory.systems.east_asian import EastAsian
from nasong.theory.generators.styles.afrobeat import Afrobeat
from nasong.theory.generators.styles.bossa_nova import BossaNova
from nasong.theory.generators.styles.salsa import Salsa
from nasong.theory.generators.styles.koto import Koto
from nasong.theory.generators.styles.celtic import Celtic


def test_global_music():
    print("Testing Global Music Expansion...")

    # Africa
    scale = African.pentatonic("C4")
    print(f"African Pentatonic: {[i.semitones for i in scale.intervals]}")
    prog = Afrobeat.polyrhythmic_groove("C4")
    print(f"Afrobeat Prog: {[c.name for c in prog.chords]}")

    # Latin America
    bossa = BossaNova.standard_progression("C4")
    print(f"Bossa Nova: {[c.name for c in bossa.chords]}")
    salsa = Salsa.montuno_progression("G4")
    print(f"Salsa Montuno: {[c.name for c in salsa.chords]}")

    # East Asia
    yo = EastAsian.yo_scale("D4")
    print(f"Yo Scale: {[i.semitones for i in yo.intervals]}")
    koto = Koto.traditional_motif("D4")
    # Koto returns a progression of single-note chords
    print(f"Koto Motif: {[c.root.name for c in koto.chords]}")

    # Europe
    celtic = Celtic.dorian_tune("D4")
    print(f"Celtic Dorian: {[c.name for c in celtic.chords]}")


if __name__ == "__main__":
    test_global_music()
