#!/usr/bin/env python

import numpy as np
import random

from marimba_samples import Marimba
from audio import Audio


def random_cents_fifths():
    total_duration_seconds = 120
    notes_per_second = 1.2

    marimba = Marimba()
    audio = Audio(total_duration_seconds)

    total_duration_samples = audio.sample_rate * total_duration_seconds

    midi_numbers = marimba.get_midi_numbers(cents=True)[:-4000]

    ticks = int(total_duration_seconds * notes_per_second)
    for _ in range(ticks):
        print _, 'of', ticks

        start = np.random.randint(audio.n_samples - 100000)
        midi_number = np.random.choice(midi_numbers)
        sample = marimba.get_note(midi_number)
        audio.add(start, sample)

        max_delay = .3 * audio.sample_rate
        end_minus_padding = audio.n_samples - (max_delay * 15)
        if start < end_minus_padding:
            delay = audio.sample_rate * np.random.normal(.12, .017)
            for _ in range(4):
                start += delay
                echo_start = int(start)
                midi_number += 7.02
                echo_sample = marimba.get_note(midi_number)
                audio.add(echo_start, echo_sample)

    audio.write_wav('random-uniform')


def random_notes():
    """Chose a random note every 6250 samples"""
    marimba = Marimba()
    audio = Audio(30)
    len_audio = len(audio) - 50000

    tick = 0
    while tick < len_audio:
        note = marimba.get_random_note()
        audio.add(tick, note)
        tick += 6250

    audio.write_wav('samples-test')


def random_notes_2():
    """Choose 100 random notes and put them in random locations throughout the duration of the audio containing the original samples"""
    marimba = Marimba()
    audio = Audio(30)
    len_audio = len(audio) - 50000
    n_notes = 100
    for i in range(n_notes):
        print '\t', i, 'of', n_notes
        note = marimba.get_random_note()
        start = np.random.randint(len_audio)

        audio.add(start, note)

    audio.write_wav('samples-test')


def microtonal_experiment_1():
    print 'Running microtonal_experiment_1...'
    marimba = Marimba()

    audio = Audio(30)

    # the marimba recordings have a duration of 50000 samples.
    # Don't start a recording within 50000 samples of the end of the audio
    len_audio = len(audio) - 50000

    print len_audio

    n_notes = 100
    for i in range(n_notes):
        print '\t', i, 'of', n_notes
        note = marimba.get_random_note(cents=True)
        start = np.random.randint(len_audio)

        audio.add(start, note)

    audio.write_wav('samples-test')
    print 'Done running microtonal_experiment_1.'


def et_experiment(pitches_per_halfstep=3, notes_per_second=2.5, duration_seconds=30):
    print 'Running et_experiment...'
    marimba = Marimba()

    audio = Audio(duration_seconds)
    max_start = len(audio) - 50000

    min_note = 36.0
    max_note = 96.0
    n_halfsteps = max_note - min_note
    n_pitches = n_halfsteps * pitches_per_halfstep
    midi_notes = np.linspace(36, 96, n_pitches, endpoint=False)

    print midi_notes

    n_notes = int(duration_seconds * notes_per_second)
    for i in range(n_notes):
        print '\t', i, 'of', n_notes

        midi_note = np.random.choice(midi_notes)
        note = marimba.get_note(midi_note)
        start = np.random.randint(max_start)

        audio.add(start, note)

    audio.write_wav('et-experiment')
    print 'Done running et_experiment.'


def block_chord_experiment(pitches_per_halfstep=3, duration_seconds=60):
    print 'Running block_chord_experiment...'
    marimba = Marimba()

    audio = Audio(duration_seconds)
    max_start = len(audio) - 50000

    min_note = 36.0
    max_note = 96.0
    n_halfsteps = max_note - min_note
    n_pitches = n_halfsteps * pitches_per_halfstep
    midi_notes = np.linspace(36, 96, n_pitches, endpoint=False)

    tick = 0
    while tick < max_start:
        for _ in range(3):
            midi_note = np.random.choice(midi_notes)
            note = marimba.get_note(midi_note)
            audio.add(tick, note)
        tick += int(round((audio.sample_rate * 0.85)))

    audio.write_wav('block-chords')
    print 'Done running block_chord_experiment.'


def common_tone_chord_experiment(pitches_per_halfstep=3, duration_seconds=60):
    print 'Running common_tone_chord_experiment...'
    marimba = Marimba()

    audio = Audio(duration_seconds)
    max_start = len(audio) - 100000

    min_note = 36.0
    max_note = 96.0
    n_halfsteps = max_note - min_note
    n_pitches = n_halfsteps * pitches_per_halfstep
    midi_notes = list(np.linspace(36, 96, n_pitches, endpoint=False))

    chord = [np.random.choice(midi_notes) for _ in range(4)]
    chord.sort()

    width = (pitches_per_halfstep * 2) - 1
    tick = 0
    while tick < max_start:
        print chord

        n_notes_to_change = np.random.choice([1, 1, 2])
        notes_to_change = random.sample(chord, n_notes_to_change)
        for midi_note in notes_to_change:
            index_in_chord = chord.index(midi_note)

            index_in_midi_notes = midi_notes.index(midi_note)

            lowest = max(index_in_midi_notes - width, 0)
            highest = min(index_in_midi_notes + width, len(midi_notes) - 1)
            new_index_in_midi_notes = np.random.choice(range(lowest, highest))
            midi_note = midi_notes[new_index_in_midi_notes]
            chord[index_in_chord] = midi_note

        for midi_note in chord:
            note = marimba.get_note(midi_note)
            tick += int(audio.sample_rate * 0.12)
            audio.add(tick, note)

        tick += int(round((audio.sample_rate * 0.85)))

    audio.write_wav('common-tone-chords')
    print 'Done running common_tone_chord_experiment.'


if __name__ == '__main__':
    # microtonal_experiment_1()
    # random_notes_2()
    # random_notes()
    # random_cents_fifths()
    # et_experiment(3, 1.8, 120)
    common_tone_chord_experiment()
