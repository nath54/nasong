from nasong.theory.systems.african import African
from nasong.theory.systems.east_asian import EastAsian
from nasong.theory.generators.styles.afrobeat import Afrobeat
from nasong.theory.generators.styles.bossa_nova import BossaNova
from nasong.theory.generators.styles.salsa import Salsa
from nasong.theory.generators.styles.koto import Koto
from nasong.theory.generators.styles.celtic import Celtic
from nasong.theory.generators.styles.jazz import Jazz
from nasong.theory.generators.styles.edm import EDM
from nasong.theory.generators.styles.lofi import Lofi


def demo_global_styles():
    print("=== Exploring Global Music Styles ===\n")

    print("--- Africa ---")
    scale = African.pentatonic("C4")
    print(f"African Pentatonic: {[i.semitones for i in scale.intervals]}")
    prog = Afrobeat.polyrhythmic_groove("C4")
    print(f"Afrobeat Groove (Chords): {[c.name for c in prog.chords]}")

    print("\n--- Latin America ---")
    bossa = BossaNova.standard_progression("C4")
    print(f"Bossa Nova: {[c.name for c in bossa.chords]}")
    salsa = Salsa.montuno_progression("G4")
    print(f"Salsa Montuno: {[c.name for c in salsa.chords]}")

    print("\n--- East Asia ---")
    yo = EastAsian.yo_scale("D4")
    print(f"Yo Scale: {[i.semitones for i in yo.intervals]}")
    koto = Koto.traditional_motif("D4")
    print(f"Koto Motif: {[c.root.name for c in koto.chords]}")

    print("\n--- Europe ---")
    celtic = Celtic.dorian_tune("D4")
    print(f"Celtic Dorian Tune: {[c.name for c in celtic.chords]}")

    print("\n--- Modern Styles ---")
    print(f"Jazz ii-V-I: {[c.name for c in Jazz.ii_V_I('C4').chords]}")
    print(f"EDM Epic: {[c.root.name for c in EDM.epic_chords('F4').chords]}")
    print(f"Lofi Chill: {[c.root.name for c in Lofi.chill_progression('Db4').chords]}")


if __name__ == "__main__":
    demo_global_styles()
