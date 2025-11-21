#
### Import Modules. ###
#
import lib_value as lv
import lib_song as ls
import lib_config as lc
#
from lib_ext_instr_1_bowed_strings import Violin, Cello

#
### Specify duration (in seconds). ###
#
duration: float = 15.0

#
### Define the song graph. ###
#
def song(time: lv.Value) -> lv.Value:

    #
    ### 1. Violin Melody (Bach-ish D Minor) ###
    #
    # D5 = 587.33, F5 = 698.46, E5 = 659.25, A4 = 440.0, C#5 = 554.37
    #
    violin_notes = [
        # Phrase 1
        (587.33, 0.5), (698.46, 0.25), (659.25, 0.25), (587.33, 0.5), (440.0, 0.5), # D F E D A
        (554.37, 0.5), (440.0, 0.5), (587.33, 1.0), # C# A D
        # Phrase 2
        (698.46, 0.5), (880.00, 0.25), (783.99, 0.25), (698.46, 0.5), (587.33, 0.5), # F A G F D
        (659.25, 0.5), (554.37, 0.5), (587.33, 1.0), # E C# D
        # Phrase 3 (Faster)
        (587.33, 0.25), (659.25, 0.25), (698.46, 0.25), (783.99, 0.25), (880.00, 0.5), (440.0, 0.5),
        (698.46, 0.25), (659.25, 0.25), (587.33, 0.25), (554.37, 0.25), (587.33, 1.5)
    ]
    
    violin_seq = lv.SimpleMelody(
        time,
        instrument_factory=Violin,
        notes=violin_notes,
        start_time=0.5,
        gap=0.0
    )

    #
    ### 2. Cello Accompaniment (Counterpoint) ###
    #
    # D3 = 146.83, A2 = 110.0, F3 = 174.61, G3 = 196.0, A3 = 220.0
    #
    cello_notes = [
        # Phrase 1
        (146.83, 1.0), (174.61, 0.5), (196.00, 0.5), # D F G
        (220.00, 1.0), (146.83, 1.0), # A D
        # Phrase 2
        (174.61, 1.0), (146.83, 0.5), (130.81, 0.5), # F D C
        (110.00, 1.0), (146.83, 1.0), # A D
        # Phrase 3
        (146.83, 0.5), (220.00, 0.5), (174.61, 0.5), (130.81, 0.5),
        (110.00, 1.0), (146.83, 2.0)
    ]
    
    cello_seq = lv.SimpleMelody(
        time,
        instrument_factory=Cello,
        notes=cello_notes,
        start_time=0.5,
        gap=0.0
    )

    #
    ### Mix ###
    #
    return lv.Sum(violin_seq, cello_seq)

