from music_encoders import MidiPerformanceEncoder

# min and max from a piano range, see http://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
encoder = MidiPerformanceEncoder(100, 0, 21, 108)


def printevents(l_events):
    results = []
    for e in l_events:
        # my_encoder.encoder._encoding.decode_event(131)
        decoded = encoder._encoding.decode_event(e)
        event_type_str = None
        if decoded.event_type == decoded.NOTE_ON:
            event_type_str = "Note ON"
        elif decoded.event_type == decoded.NOTE_OFF:
            event_type_str = "Note OFF"
        elif decoded.event_type == decoded.TIME_SHIFT:
            event_type_str = "Time Shift"
        results.append('PerformanceEvent(%r, %r)' %
                       (event_type_str, decoded.event_value))
    return results
