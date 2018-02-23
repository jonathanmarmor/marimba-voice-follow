#!/usr/bin/env python

import numpy as np

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
    len_audio = audio.len() - 50000

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
    len_audio = audio.len() - 50000
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
    len_audio = audio.len() - 50000

    n_notes = 100
    for i in range(n_notes):
        print '\t', i, 'of', n_notes
        note = marimba.get_random_note(cents=True)
        start = np.random.randint(len_audio)

        audio.add(start, note)

    audio.write_wav('samples-test')
    print 'Done running microtonal_experiment_1.'


if __name__ == '__main__':
    # microtonal_experiment_1()
    # random_notes_2()
    # random_notes()
    random_cents_fifths()
