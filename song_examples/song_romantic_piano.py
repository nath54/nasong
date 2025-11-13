#
### Import Modules. ###
#
import lib_value as lv
#
from lib_ext_instr_1 import PianoNote


#
### Specify duration (in seconds). ###
#
duration: float = 30.0


#
### Song description: Romantic Piano Ballad ###
#
def song(time: lv.Value) -> lv.Value:
    """
    A romantic, melodic piano piece with a beautiful chord progression.
    Key: C major / A minor
    Progression: C - Am - F - G - C - Am - Dm - G
    """

    # Chord progression (frequencies in Hz)
    # Measure 1-2: C major (C4, E4, G4)
    c_chord = lv.Sum(
        PianoNote(time, 261.63, 0.0, 3.5),    # C4
        PianoNote(time, 329.63, 0.1, 3.4),    # E4
        PianoNote(time, 392.00, 0.2, 3.3)     # G4
    )

    # Measure 3-4: A minor (A3, C4, E4)
    am_chord = lv.Sum(
        PianoNote(time, 220.00, 4.0, 3.5),    # A3
        PianoNote(time, 261.63, 4.1, 3.4),    # C4
        PianoNote(time, 329.63, 4.2, 3.3)     # E4
    )

    # Measure 5-6: F major (F3, A3, C4)
    f_chord = lv.Sum(
        PianoNote(time, 174.61, 8.0, 3.5),    # F3
        PianoNote(time, 220.00, 8.1, 3.4),    # A3
        PianoNote(time, 261.63, 8.2, 3.3)     # C4
    )

    # Measure 7-8: G major (G3, B3, D4)
    g_chord = lv.Sum(
        PianoNote(time, 196.00, 12.0, 3.5),   # G3
        PianoNote(time, 246.94, 12.1, 3.4),   # B3
        PianoNote(time, 293.66, 12.2, 3.3)    # D4
    )

    # Measure 9-10: C major (higher voicing)
    c_chord2 = lv.Sum(
        PianoNote(time, 261.63, 16.0, 3.5),   # C4
        PianoNote(time, 329.63, 16.1, 3.4),   # E4
        PianoNote(time, 392.00, 16.2, 3.3)    # G4
    )

    # Measure 11-12: A minor
    am_chord2 = lv.Sum(
        PianoNote(time, 220.00, 20.0, 3.5),   # A3
        PianoNote(time, 261.63, 20.1, 3.4),   # C4
        PianoNote(time, 329.63, 20.2, 3.3)    # E4
    )

    # Measure 13-14: D minor (D3, F3, A3)
    dm_chord = lv.Sum(
        PianoNote(time, 146.83, 24.0, 3.5),   # D3
        PianoNote(time, 174.61, 24.1, 3.4),   # F3
        PianoNote(time, 220.00, 24.2, 3.3)    # A3
    )

    # Measure 15-16: G major (resolution)
    g_chord2 = lv.Sum(
        PianoNote(time, 196.00, 28.0, 3.5),   # G3
        PianoNote(time, 246.94, 28.1, 3.4),   # B3
        PianoNote(time, 293.66, 28.2, 3.3)    # D4
    )

    # Add melody on top
    melody = lv.Sum(
        PianoNote(time, 523.25, 0.5, 1.0),    # C5
        PianoNote(time, 493.88, 2.0, 1.0),    # B4
        PianoNote(time, 440.00, 3.5, 1.0),    # A4
        PianoNote(time, 523.25, 5.0, 1.5),    # C5
        PianoNote(time, 587.33, 7.0, 1.0),    # D5
        PianoNote(time, 523.25, 8.5, 1.5),    # C5
        PianoNote(time, 493.88, 10.5, 1.0),   # B4
        PianoNote(time, 440.00, 12.0, 2.0),   # A4
        PianoNote(time, 392.00, 14.5, 1.5),   # G4
        PianoNote(time, 523.25, 16.5, 1.0),   # C5
        PianoNote(time, 493.88, 18.0, 1.0),   # B4
        PianoNote(time, 440.00, 19.5, 1.5),   # A4
        PianoNote(time, 392.00, 21.5, 1.0),   # G4
        PianoNote(time, 349.23, 23.0, 1.5),   # F4
        PianoNote(time, 329.63, 25.0, 2.0),   # E4
        PianoNote(time, 293.66, 27.5, 1.5),   # D4
        PianoNote(time, 261.63, 29.0, 2.0)    # C4 (ending)
    )

    # Combine all elements
    final = lv.Sum(
        c_chord, am_chord, f_chord, g_chord,
        c_chord2, am_chord2, dm_chord, g_chord2,
        lv.Product(melody, lv.Constant(0.6))
    )

    return lv.Product(final, lv.Constant(0.5))
