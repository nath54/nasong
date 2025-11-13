#
### Import Modules. ###
#
import lib_value as lv
#
from lib_ext_instr_1 import DeepBass
#
from lib_ext_percus_1 import KickDrum, SnareDrum, HiHat, CrashCymbal


#
### Specify duration (in seconds). ###
#
duration: float = 16.0


#
### Song description: Drum Beat with Bass ###
#
def song(time: lv.Value) -> lv.Value:
    """
    A powerful drum beat with kick, snare, hi-hats, and deep bass.
    Pattern: 4/4 time at 120 BPM (0.5 seconds per beat)
    """

    # Kick drum pattern (on beats 1 and 3, plus some variations)
    kicks: list[lv.Value] = []
    for bar in range(8):
        base = bar * 2.0
        kicks.append(KickDrum(time, base + 0.0))      # Beat 1
        kicks.append(KickDrum(time, base + 1.0))      # Beat 3
        if bar % 2 == 1:
            kicks.append(KickDrum(time, base + 1.75))  # Variation

    # Snare drum pattern (on beats 2 and 4)
    snares: list[lv.Value] = []
    for bar in range(8):
        base = bar * 2.0
        snares.append(SnareDrum(time, base + 0.5))    # Beat 2
        snares.append(SnareDrum(time, base + 1.5))    # Beat 4

    # Hi-hat pattern (eighth notes)
    hihats: list[lv.Value] = []
    for i in range(64):  # 16 seconds * 4 eighth notes per second
        t = i * 0.25
        is_open = (i % 4 == 2)  # Open hi-hat on upbeats occasionally
        hihats.append(HiHat(time, t, open=is_open))

    # Crash cymbals (at the beginning and middle)
    crashes = [
        CrashCymbal(time, 0.0),
        CrashCymbal(time, 8.0),
    ]

    #
    freq_factor_bass = 2.0
    duration_factor_bass = 6.0

    # Bass line (following kick drum mostly)
    bass_notes = [
        # Bar 1
        DeepBass(time, freq_factor_bass * 55.00, 0.0, duration_factor_bass * 0.5),      # A1
        DeepBass(time, freq_factor_bass * 55.00, 1.0, duration_factor_bass * 0.5),

        # Bar 2
        DeepBass(time, freq_factor_bass * 55.00, 2.0, duration_factor_bass * 0.5),
        DeepBass(time, freq_factor_bass * 65.41, 3.0, duration_factor_bass * 0.5),      # C2
        DeepBass(time, freq_factor_bass * 73.42, 3.5, duration_factor_bass * 0.4),      # D2

        # Bar 3
        DeepBass(time, freq_factor_bass * 55.00, 4.0, duration_factor_bass * 0.5),
        DeepBass(time, freq_factor_bass * 55.00, 5.0, duration_factor_bass * 0.5),

        # Bar 4
        DeepBass(time, freq_factor_bass * 55.00, 6.0, duration_factor_bass * 0.5),
        DeepBass(time, freq_factor_bass * 65.41, 7.0, duration_factor_bass * 0.5),
        DeepBass(time, freq_factor_bass * 73.42, 7.5, duration_factor_bass * 0.4),

        # Bar 5
        DeepBass(time, freq_factor_bass * 49.00, 8.0, duration_factor_bass * 0.5),      # G1
        DeepBass(time, freq_factor_bass * 49.00, 9.0, duration_factor_bass * 0.5),

        # Bar 6
        DeepBass(time, freq_factor_bass * 49.00, 10.0, duration_factor_bass * 0.5),
        DeepBass(time, freq_factor_bass * 55.00, 11.0, duration_factor_bass * 0.5),     # A1
        DeepBass(time, freq_factor_bass * 61.74, 11.5, duration_factor_bass * 0.4),     # B1

        # Bar 7
        DeepBass(time, freq_factor_bass * 55.00, 12.0, duration_factor_bass * 0.5),
        DeepBass(time, freq_factor_bass * 55.00, 13.0, duration_factor_bass * 0.5),

        # Bar 8 (ending)
        DeepBass(time, freq_factor_bass * 55.00, 14.0, duration_factor_bass * 0.5),
        DeepBass(time, freq_factor_bass * 73.42, 15.0, duration_factor_bass * 0.5),
        DeepBass(time, freq_factor_bass * 55.00, 15.5, duration_factor_bass * 0.5),
    ]

    # Combine all drum elements
    drums = lv.Sum(
        *kicks,
        *snares,
        *hihats,
        *crashes
    )

    bass = lv.Sum(*bass_notes)

    # Mix everything together
    final = lv.Sum(
        lv.Product(drums, lv.Constant(0.85)),
        lv.Product(bass, lv.Constant(1.0))
    )

    return lv.Product(final, lv.Constant(0.7))
