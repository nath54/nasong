from .structures.note import Note as NoteEvent
from .structures.chord import Chord
from .structures.progression import Progression

from typing import List


def expand(obj) -> List[NoteEvent]:
    """
    Flatten a musical object into a list of NoteEvents.
    """
    if isinstance(obj, NoteEvent):
        return [obj]
    if isinstance(obj, Chord):
        # Chord.notes returns list of NoteEvents (simultaneous)
        # But for sequencing, checks handling simultaneous notes.
        # This function returns a list of events. They might need start times?
        # NoteEvent doesn't have absolute start time, only duration.
        # A Chord is a simultaneity.
        # If we return a list, it implies sequence?
        # Or just a collection.
        return obj.notes
    if isinstance(obj, Progression):
        # A progression is a sequence of Chords.
        # We can't just return a flat list of notes without timing info if we want to preserve structure?
        # But expand usually implies "get all notes".
        events = []
        for chord in obj.chords:
            events.extend(chord.notes)
        return events
    return []


def track_from_progression(
    progression: Progression, start_time: float = 0.0
) -> List[tuple[float, NoteEvent]]:
    """
    Convert a progression into a distinct list of (absolute_start_time, NoteEvent).
    useful for feeding into a Sequencer.
    """
    track = []
    current_time = start_time

    for chord in progression.chords:
        # All notes in chord start at current_time
        duration_quarters = chord.duration.value
        for note in chord.notes:
            track.append((current_time, note))

        current_time += duration_quarters

    return track


def render(obj, time_value, instrument_factory, bpm: float = 120.0):
    """
    Create a core.Value (Sequencer) from a theory object.

    Args:
        obj: Progression, Chord, or Note.
        time_value: The global time Value.
        instrument_factory: A function (freq, duration_sec, velocity) -> Value
                            OR using the signature expected by Sequencer:
                            (time_val, freq, start_time, duration, [extras]) -> Value
        bpm: Beats per minute (assumes Quarter note = 1 beat).
    """
    from nasong.core.values.single_itms_ops.value_sequencer import Sequencer
    from nasong.core.values.basic.value_identity import Identity

    if time_value is None:
        time_value = Identity()

    # Calculate duration of a quarter note in seconds
    # BPM = quarters per minute
    # quarters per second = BPM / 60
    # seconds per quarter = 60 / BPM
    sec_per_quarter = 60.0 / bpm

    track_data = []  # (freq, start_sec, dur_sec, velocity)

    if isinstance(obj, Progression):
        events = track_from_progression(obj)
        # events is list of (start_quarters, NoteEvent)
        for start_q, note in events:
            start_sec = start_q * sec_per_quarter
            dur_sec = note.duration.value * sec_per_quarter
            freq = note.pitch.to_hz()
            vel = note.velocity
            track_data.append((freq, start_sec, dur_sec, vel))

    elif isinstance(obj, Chord):
        # Chord starts at 0 relative to render?
        for note in obj.notes:
            start_sec = 0.0
            dur_sec = note.duration.value * sec_per_quarter
            freq = note.pitch.to_hz()
            vel = note.velocity
            track_data.append((freq, start_sec, dur_sec, vel))

    elif isinstance(obj, NoteEvent):
        start_sec = 0.0
        dur_sec = obj.duration.value * sec_per_quarter
        freq = obj.pitch.to_hz()
        vel = obj.velocity
        track_data.append((freq, start_sec, dur_sec, vel))

    # Sequencer expects note_data_list.
    # The default Sequencer uses factory(time, *args).
    # so args will be (freq, start_sec, dur_sec, vel) if we pass that.

    # We need to adapt track_data to what Sequencer expects.
    # Sequencer expects note_data_list items to match factory args.
    # Usually: (freq, start, duration) are standard.
    # Let's clean up track_data to standard format: (freq, start, duration)
    # If instrument_factory supports velocity, we append it.

    # Let's assume factory supports velocity.

    return Sequencer(
        time_value, instrument_factory=instrument_factory, note_data_list=track_data
    )
