#
### Import Modules. ###
#
import lib_value as lv
#
from lib_ext_synth_1 import SynthBass, SynthPad, SynthLead

#
### Specify duration (in seconds). ###
#
duration: float = 21.0


#
### Song description: Electronic Synth Lead ###
#
def song(time: lv.Value) -> lv.Value:
    """
    An electronic music piece with synth leads, bass, and pads.
    Key: A minor
    BPM: 128
    """

    # Bass line (repeating pattern)
    bass_pattern: list[lv.Value] = [
        SynthBass(time, 110.00, 0.0, 0.4),    # A2
        SynthBass(time, 110.00, 0.5, 0.4),
        SynthBass(time, 146.83, 1.0, 0.4),    # D3
        SynthBass(time, 110.00, 1.5, 0.4),

        SynthBass(time, 98.00, 2.0, 0.4),     # G2
        SynthBass(time, 98.00, 2.5, 0.4),
        SynthBass(time, 110.00, 3.0, 0.4),
        SynthBass(time, 123.47, 3.5, 0.4),    # B2

        # Repeat pattern
        SynthBass(time, 110.00, 4.0, 0.4),
        SynthBass(time, 110.00, 4.5, 0.4),
        SynthBass(time, 146.83, 5.0, 0.4),
        SynthBass(time, 110.00, 5.5, 0.4),

        SynthBass(time, 98.00, 6.0, 0.4),
        SynthBass(time, 98.00, 6.5, 0.4),
        SynthBass(time, 110.00, 7.0, 0.4),
        SynthBass(time, 123.47, 7.5, 0.4),

        # Continue for full 20 seconds
        SynthBass(time, 110.00, 8.0, 0.4),
        SynthBass(time, 110.00, 8.5, 0.4),
        SynthBass(time, 146.83, 9.0, 0.4),
        SynthBass(time, 110.00, 9.5, 0.4),

        SynthBass(time, 98.00, 10.0, 0.4),
        SynthBass(time, 98.00, 10.5, 0.4),
        SynthBass(time, 110.00, 11.0, 0.4),
        SynthBass(time, 123.47, 11.5, 0.4),

        SynthBass(time, 110.00, 12.0, 0.4),
        SynthBass(time, 110.00, 12.5, 0.4),
        SynthBass(time, 146.83, 13.0, 0.4),
        SynthBass(time, 110.00, 13.5, 0.4),

        SynthBass(time, 98.00, 14.0, 0.4),
        SynthBass(time, 98.00, 14.5, 0.4),
        SynthBass(time, 110.00, 15.0, 0.4),
        SynthBass(time, 123.47, 15.5, 0.4),

        SynthBass(time, 110.00, 16.0, 0.4),
        SynthBass(time, 110.00, 16.5, 0.4),
        SynthBass(time, 146.83, 17.0, 0.4),
        SynthBass(time, 110.00, 17.5, 0.4),

        SynthBass(time, 98.00, 18.0, 0.4),
        SynthBass(time, 110.00, 18.5, 0.8),
    ]

    # Atmospheric pads (chord progression)
    pads = [
        SynthPad(time, 220.00, 0.0, 8.0),     # A3
        SynthPad(time, 261.63, 0.0, 8.0),     # C4
        SynthPad(time, 329.63, 0.0, 8.0),     # E4

        SynthPad(time, 196.00, 8.0, 8.0),     # G3
        SynthPad(time, 246.94, 8.0, 8.0),     # B3
        SynthPad(time, 293.66, 8.0, 8.0),     # D4

        SynthPad(time, 220.00, 16.0, 4.0),
        SynthPad(time, 261.63, 16.0, 4.0),
        SynthPad(time, 329.63, 16.0, 4.0),
    ]

    # Lead melody
    lead_melody = [
        SynthLead(time, 440.00, 2.0, 1.5),    # A4
        SynthLead(time, 493.88, 3.5, 1.0),    # B4
        SynthLead(time, 523.25, 4.5, 1.5),    # C5
        SynthLead(time, 493.88, 6.0, 1.0),    # B4
        SynthLead(time, 440.00, 7.0, 2.0),    # A4

        SynthLead(time, 392.00, 10.0, 1.5),   # G4
        SynthLead(time, 440.00, 11.5, 1.0),   # A4
        SynthLead(time, 493.88, 12.5, 1.5),   # B4
        SynthLead(time, 523.25, 14.0, 1.0),   # C5
        SynthLead(time, 587.33, 15.0, 2.0),   # D5

        SynthLead(time, 523.25, 17.0, 1.5),   # C5
        SynthLead(time, 493.88, 18.5, 2.0),   # B4
    ]

    # Combine all elements
    bass = lv.Sum(*bass_pattern)
    pad = lv.Sum(*pads)
    _lead = lv.Sum(*lead_melody)

    final = lv.Sum(
        lv.Product(bass, lv.Constant(0.9)),
        lv.Product(pad, lv.Constant(0.6)),
        lv.Product(_lead, lv.Constant(1.0))
    )

    return lv.Product(final, lv.Constant(0.65))
