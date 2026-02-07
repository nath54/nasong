#
### Import Modules. ###
#
from nasong.trainable.instruments import (
    TrainableSawtoothSynth,
    TrainableSquareSynth,
    TrainableSineSynth,
    TrainableKick,
    TrainableSnare,
    TrainableHiHat,
    TrainablePlucked,
    TrainablePiano,
    TrainableBowed,
    TrainablePad,
    TrainableBass,
    TrainableNamedExamples,
)

#
### INSTRUMENT REGISTRY ###
#

# Dictionary of all available trainable instruments
TRAINABLE_INSTRUMENTS = {
    # Synthesis
    "sawtooth": TrainableSawtoothSynth,
    "square": TrainableSquareSynth,
    "sine": TrainableSineSynth,
    # Percussion
    "kick": TrainableKick,  # Note: kick doesn't take frequency
    "snare": TrainableSnare,  # Note: snare doesn't take frequency
    "hihat_closed": lambda t, f, st, d: TrainableHiHat(t, st, is_open=False),
    "hihat_open": lambda t, f, st, d: TrainableHiHat(t, st, is_open=True),
    # Melodic
    "plucked": TrainablePlucked,
    "piano": TrainablePiano,
    "bowed": TrainableBowed,
    # Atmospheric
    "pad": TrainablePad,
    # Bass
    # Bass
    "bass": TrainableBass,
    # Examples
    "named_fm": TrainableNamedExamples,
}


def get_trainable_instrument(instrument_name: str):
    """
    Get a trainable instrument blueprint by name.

    Args:
        instrument_name: Name of instrument from TRAINABLE_INSTRUMENTS

    Returns:
        Instrument blueprint function
    """
    if instrument_name not in TRAINABLE_INSTRUMENTS:
        raise ValueError(
            f"Unknown instrument: {instrument_name}. Available: {list(TRAINABLE_INSTRUMENTS.keys())}"
        )

    return TRAINABLE_INSTRUMENTS[instrument_name]
