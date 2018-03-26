#!/usr/bin/env python

import math
import random

import numpy as np

from marimba_samples import Marimba
from audio import Audio
from utils import random_from_range, ratio_to_cents


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
            tick += int(audio.sample_rate * 0.08)
            audio.add(tick, note)

        tick += int(round((audio.sample_rate * 0.85)))

    audio.write_wav('common-tone-chords')
    print 'Done running common_tone_chord_experiment.'


def pulse(audio, instrument, midi_note=60.0, duration_seconds=1.0, random_mute_threshhold=1.0):
    duration_samples = int(round(duration_seconds * audio.sample_rate))
    note =  instrument.get_note(midi_note)

    samples = 0
    while samples < len(audio) - (len(note) * 2):
        if np.random.random() < random_mute_threshhold:
            audio.add(samples, note)
        samples += duration_samples


def multiple_tempos():
    print 'Running multiple_tempos...'
    marimba = Marimba()
    audio = Audio(120)

    pulse(audio, marimba, 41.0,  1.5)  # 1 -- 0
    pulse(audio, marimba, 48.02, 1.4)  # 3 -- 702

    pulse(audio, marimba, 53.0,  1.3)  # 1 -- 0
    pulse(audio, marimba, 56.86, 1.2)  # 5 -- 386
    pulse(audio, marimba, 60.02, 1.1)  # 3 -- 702
    pulse(audio, marimba, 62.69, 1.0)  # 7 -- 969

    pulse(audio, marimba, 65.0,  0.9) # 1 -- 0
    pulse(audio, marimba, 67.04, 0.8) # 9 -- 204
    pulse(audio, marimba, 68.86, 0.7) # 5 -- 386
    pulse(audio, marimba, 70.51, 0.6) # 11 - 551
    pulse(audio, marimba, 72.02, 0.5) # 3 -- 702
    pulse(audio, marimba, 73.41, 0.4) # 13 -- 841
    pulse(audio, marimba, 74.69, 0.3) # 7 -- 969
    pulse(audio, marimba, 75.88, 0.2) # 15 -- 1088

    # pulse(audio, marimba, 77.0, 0.1) # 1 -- 0

    audio.write_wav('multiple-tempos')
    print 'Done running multiple_tempos.'


def multiple_tempos_muting():
    print 'Running multiple_tempos_muting...'
    marimba = Marimba()
    audio = Audio(120)

    pulse(audio, marimba, 41.0,  1.5, 14.0 / 15)  # 1 -- 0
    pulse(audio, marimba, 48.02, 1.4, 13.0 / 15)  # 3 -- 702

    pulse(audio, marimba, 53.0,  1.3, 12.0 / 15)  # 1 -- 0
    pulse(audio, marimba, 56.86, 1.2, 11.0 / 15)  # 5 -- 386
    pulse(audio, marimba, 60.02, 1.1, 10.0 / 15)  # 3 -- 702
    pulse(audio, marimba, 62.69, 1.0,  9.0 / 15)  # 7 -- 969

    pulse(audio, marimba, 65.0,  0.9,  8.0 / 15) # 1 -- 0
    pulse(audio, marimba, 67.04, 0.8,  7.0 / 15) # 9 -- 204
    pulse(audio, marimba, 68.86, 0.7,  6.0 / 15) # 5 -- 386
    pulse(audio, marimba, 70.51, 0.6,  5.0 / 15) # 11 - 551
    pulse(audio, marimba, 72.02, 0.5,  4.0 / 15) # 3 -- 702
    pulse(audio, marimba, 73.41, 0.4,  3.0 / 15) # 13 -- 841
    pulse(audio, marimba, 74.69, 0.3,  2.0 / 15) # 7 -- 969
    pulse(audio, marimba, 75.88, 0.2,  1.0 / 15) # 15 -- 1088

    audio.write_wav('multiple-tempos-muting')
    print 'Done running multiple_tempos_muting.'


def get_slice_of_harmonic_series(lowest_harmonic, n_harmonics, lowest_midi_note=41.0):
    harmonics = np.linspace(lowest_harmonic, n_harmonics + lowest_harmonic - 1, n_harmonics)

    lowest_harmonic_cents = ratio_to_cents(lowest_harmonic, 1.0, round_decimal_places=None)

    midi_notes = []
    for h in harmonics:
        cents = ratio_to_cents(h, 1.0, round_decimal_places=None)
        cents = cents - lowest_harmonic_cents
        midi_note = cents / 100
        midi_note += lowest_midi_note
        midi_note = round(midi_note, 2)
        midi_notes.append(midi_note)

    return midi_notes


def multiple_tempos_refactored():
    print 'Running multiple_tempos_refactored...'
    marimba = Marimba()
    audio = Audio(120)

    quarter_duration_in_seconds = 1.2

    n_parts = 23

    lowest_harmonic = 1
    n_harmonics = n_parts

    lowest_midi_note = 36.0

    pitches = get_slice_of_harmonic_series(lowest_harmonic, n_harmonics, lowest_midi_note=lowest_midi_note)

    durations = np.linspace(1.5, .2, n_parts)

    random_mute_threshholds = [n / (n_parts + 1) for n in np.linspace(n_parts, 1, n_parts)]

    print random_mute_threshholds

    for pitch, duration, random_mute_threshhold in zip(pitches, durations, random_mute_threshholds):
        duration *= quarter_duration_in_seconds
        pulse(audio, marimba, pitch, duration, random_mute_threshhold)

    audio.write_wav('multiple-tempos-muting')
    print 'Done running multiple_tempos_refactored.'


def multiple_random_tempos():
    print 'Running multiple_random_tempos...'
    marimba = Marimba()
    audio = Audio(1200)

    quarter_duration_in_seconds = 1.0

    n_parts = 40

    lowest_harmonic = 2
    n_harmonics = n_parts

    lowest_midi_note = 36.0

    pitches = get_slice_of_harmonic_series(lowest_harmonic, n_harmonics, lowest_midi_note=lowest_midi_note)

    durations = random_from_range(2.0, 6.0, size=n_parts)

    print durations

    for pitch, duration in zip(pitches, durations):
        duration *= quarter_duration_in_seconds
        pulse(audio, marimba, pitch, duration, 0.75)

    audio.write_wav('multiple-random-tempos')
    print 'Done running multiple_random_tempos.'


def sections():
    print 'Running sections...'
    marimba = Marimba()

    quarter_duration_in_seconds = 1.0

    beats_per_section = 2.0
    section_duration_seconds = beats_per_section * quarter_duration_in_seconds
    n_sections = 64

    audio = Audio(int(section_duration_seconds * n_sections) + 3)

    section_duration_samples = int(round(section_duration_seconds * audio.sample_rate))

    chords = []
    for _ in range(n_sections):
        root = np.random.choice(np.linspace(0, 12, 24, endpoint=False))
        chord_type = random.choice([
            [0, 2.5, 5, 7, 9.5],
            [0, 2.5, 7, 9.5],
            [0, 3.5, 7, 10.5],
            [0, 4.5, 7],
        ])
        chord = [(p + root) % 12 for p in chord_type]
        chords.append(chord)

    start = 0
    for section_number in range(n_sections):
        chord = chords[section_number]
        lowest = 50.0
        highest = 86.0
        pitches = [p for p in np.linspace(lowest, highest, (highest - lowest) * 100, endpoint=False) if p % 12 in chord]

        end = start + section_duration_samples

        for _ in range(int(beats_per_section * random.choice([1, 2, 4, 8, 16]))):
            note_start = np.random.randint(start, end - 5000)

            midi_note = np.random.choice(pitches)

            note = marimba.get_note(midi_note)
            audio.add(note_start, note)


        start += section_duration_samples

    audio.write_wav('sections')
    print 'Done running sections.'


if __name__ == '__main__':
    # microtonal_experiment_1()
    # random_notes_2()
    # random_notes()
    # random_cents_fifths()
    # et_experiment(3, 1.8, 120)
    # common_tone_chord_experiment()
    # multiple_tempos()
    # multiple_tempos_muting()
    # multiple_tempos_refactored()
    multiple_random_tempos()
    sections()

