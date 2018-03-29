import numpy as np


class Marimba(object):
    def __init__(self):
        print 'Loading marimba samples...'
        self._wavetable = np.load('samples/marimba/marimba-cents.npy')
        print 'Done.'

        self.min_midi_number = 36.0
        self.min_midi_number_times_100 = round(self.min_midi_number * 100)
        self.max_midi_number = 96.0

        self.n_notes_cents = len(self._wavetable)
        self.n_notes = self.n_notes_cents / 100

    def get_note(self, midi_number):
        index = int(round(midi_number * 100) - self.min_midi_number_times_100)
        return self._wavetable[index]

    def get_midi_numbers(self, cents=False):
        n_pitches = self.max_midi_number - self.min_midi_number
        if cents:
            n_pitches *= 100
        return np.linspace(self.min_midi_number, self.max_midi_number, n_pitches, endpoint=False)

    def get_random_note(self, cents=False):
        n_notes = self.n_notes
        if cents:
            n_notes = self.n_notes_cents
        note_index = np.random.randint(n_notes)
        if not cents:
            note_index *= 100
        return self._wavetable[note_index]

    def get_staccato_note(self, midi_number, new_note_duration=15000):
        new_note_duration = int(new_note_duration)
        fade_duration = int(new_note_duration / 2)

        note = self.get_note(midi_number)

        new_note = note.copy()
        new_note = new_note[:new_note_duration]
        new_note[-fade_duration:] = new_note[-fade_duration:] * np.linspace(1, 0, fade_duration)

        return new_note
