#
### Import Modules. ###
#
import lib_value as lv
#
from lib_ext_instr_1 import Strum, GuitarString

#
### Specify duration (in seconds). ###
#
duration: float = 24.0


#
### Song description: Acoustic Guitar Strumming Pattern ###
#
def song(time: lv.Value) -> lv.Value:
    """
    An acoustic guitar piece with fingerpicking and strumming.
    Key: D major
    Progression: D - A - Bm - G
    """

    # D major chord (D3, A3, D4, F#4)
    d_major_strum1 = Strum(time, [146.83, 220.00, 293.66, 369.99], 0.0, 3.0)
    d_major_strum2 = Strum(time, [146.83, 220.00, 293.66, 369.99], 1.0, 2.5)

    # A major chord (A2, E3, A3, C#4)
    a_major_strum1 = Strum(time, [110.00, 164.81, 220.00, 277.18], 3.0, 3.0)
    a_major_strum2 = Strum(time, [110.00, 164.81, 220.00, 277.18], 4.0, 2.5)

    # B minor chord (B2, F#3, B3, D4)
    bm_strum1 = Strum(time, [123.47, 185.00, 246.94, 293.66], 6.0, 3.0)
    bm_strum2 = Strum(time, [123.47, 185.00, 246.94, 293.66], 7.0, 2.5)

    # G major chord (G2, D3, G3, B3)
    g_major_strum1 = Strum(time, [98.00, 146.83, 196.00, 246.94], 9.0, 3.0)
    g_major_strum2 = Strum(time, [98.00, 146.83, 196.00, 246.94], 10.0, 2.5)

    # Second progression (same pattern)
    d_major_strum3 = Strum(time, [146.83, 220.00, 293.66, 369.99], 12.0, 3.0)
    d_major_strum4 = Strum(time, [146.83, 220.00, 293.66, 369.99], 13.0, 2.5)

    a_major_strum3 = Strum(time, [110.00, 164.81, 220.00, 277.18], 15.0, 3.0)
    a_major_strum4 = Strum(time, [110.00, 164.81, 220.00, 277.18], 16.0, 2.5)

    bm_strum3 = Strum(time, [123.47, 185.00, 246.94, 293.66], 18.0, 3.0)
    bm_strum4 = Strum(time, [123.47, 185.00, 246.94, 293.66], 19.0, 2.5)

    g_major_strum3 = Strum(time, [98.00, 146.83, 196.00, 246.94], 21.0, 3.0)
    g_major_ending = Strum(time, [98.00, 146.83, 196.00, 246.94], 22.5, 2.5)

    # Fingerpicking melody on top
    melody_notes = [
        GuitarString(time, 587.33, 0.5, 1.5, 1.2),    # D5
        GuitarString(time, 554.37, 1.5, 1.0, 1.2),    # C#5
        GuitarString(time, 493.88, 2.5, 1.5, 1.2),    # B4

        GuitarString(time, 554.37, 3.5, 1.5, 1.2),    # C#5
        GuitarString(time, 587.33, 4.5, 1.0, 1.2),    # D5
        GuitarString(time, 659.25, 5.5, 1.5, 1.2),    # E5

        GuitarString(time, 587.33, 6.5, 1.5, 1.2),    # D5
        GuitarString(time, 493.88, 7.5, 1.0, 1.2),    # B4
        GuitarString(time, 440.00, 8.5, 1.5, 1.2),    # A4

        GuitarString(time, 493.88, 9.5, 1.5, 1.2),    # B4
        GuitarString(time, 440.00, 10.5, 1.0, 1.2),   # A4
        GuitarString(time, 392.00, 11.5, 2.0, 1.2),   # G4

        # Second verse
        GuitarString(time, 587.33, 13.0, 1.5, 1.2),
        GuitarString(time, 659.25, 14.0, 1.0, 1.2),
        GuitarString(time, 587.33, 15.0, 1.5, 1.2),

        GuitarString(time, 554.37, 16.5, 1.5, 1.2),
        GuitarString(time, 493.88, 17.5, 1.0, 1.2),
        GuitarString(time, 440.00, 18.5, 1.5, 1.2),

        GuitarString(time, 493.88, 20.0, 1.5, 1.2),
        GuitarString(time, 440.00, 21.0, 1.0, 1.2),
        GuitarString(time, 392.00, 22.0, 2.5, 1.0),   # Final note
    ]

    # Combine all elements
    chords = lv.Sum(
        d_major_strum1, d_major_strum2,
        a_major_strum1, a_major_strum2,
        bm_strum1, bm_strum2,
        g_major_strum1, g_major_strum2,
        d_major_strum3, d_major_strum4,
        a_major_strum3, a_major_strum4,
        bm_strum3, bm_strum4,
        g_major_strum3, g_major_ending
    )

    melody = lv.Sum(*melody_notes)

    final = lv.Sum(
        lv.Product(chords, lv.Constant(0.7)),
        lv.Product(melody, lv.Constant(0.5))
    )

    return lv.Product(final, lv.Constant(0.6))
