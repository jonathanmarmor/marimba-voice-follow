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


def overlapping_sections():
    func_name = 'overlapping_sections'
    print 'Running {}...'.format(func_name)

    marimba = Marimba()

    quarter_duration_in_seconds = 1.0

    beats_per_harmony_section = 2.0
    beats_per_density_section = 1.7777
    n_beats = beats_per_harmony_section * beats_per_density_section * 32
    duration_in_seconds = int(n_beats * quarter_duration_in_seconds + 3)

    audio = Audio(duration_in_seconds)

    harmony_section_duration_samples = int(round((beats_per_harmony_section * quarter_duration_in_seconds) * audio.sample_rate))
    start = 0
    harmony_section_starts = []
    while start < len(audio):
        harmony_section_starts.append(start)
        start += harmony_section_duration_samples

    density_section_duration_samples = int(round((beats_per_harmony_section * quarter_duration_in_seconds) * audio.sample_rate))
    start = 0
    density_section_starts = []
    while start < len(audio):
        density_section_starts.append(start)
        start += density_section_duration_samples

    max_start = len(audio) - 500000

    chords = [
        [0, 4, 7],
        [0, 5, 9],
        [0, 4, 7],
        [0, 5, 9],
        [0, 4, 7],
        [0, 5, 9],
        [2, 5, 9, 0],
        [2, 5, 7, 10]
    ]

    lowest = 59
    highest = 62

    density_section_starts = [s for s in density_section_starts if s < max_start]
    len_density_section_starts = len(density_section_starts)
    densities = np.linspace(1, 70, len_density_section_starts)

    for density, section_start in zip(densities, density_section_starts):
        for _ in range(int(density)):
            start = np.random.randint(section_start, section_start + density_section_duration_samples)
            for tick, next_tick in zip(harmony_section_starts[:-1], harmony_section_starts[1:]):
                if tick <= start < next_tick:
                    break

            chord_index = harmony_section_starts.index(tick) % len(chords)
            chord_type = chords[chord_index]
            # pitch = random.choice([p for p in range(lowest, highest) if p % 12 in chord_type])
            pitch = random.choice([p for p in np.linspace(lowest, highest, (highest - lowest) * 100, endpoint=False) if round(p % 12, 2) in chord_type])
            note = marimba.get_note(pitch)

            audio.add(start, note)

        if random.random() < .3:
            lowest = max([36, lowest - 1])
        if random.random() < .6:
            highest = min([95, highest + 1])

    # Future TODO
    # - at one rate, increase density
    # - at another rate, increase register
    # - at another rate, move from staccatto to sustained
    # - at another rate, do something else
    # - etc

    audio.write_wav(func_name)
    print 'Done running {}.'.format(func_name)


def overlapping_sections_with_weird_modulation():
    func_name = 'overlapping_sections_with_weird_modulation'
    print 'Running {}...'.format(func_name)

    marimba = Marimba()

    quarter_duration_in_seconds = 1.0

    beats_per_harmony_section = 2.0
    beats_per_density_section = 1.7777
    n_beats = beats_per_harmony_section * beats_per_density_section * 32
    duration_in_seconds = int(n_beats * quarter_duration_in_seconds + 3)

    audio = Audio(duration_in_seconds)

    harmony_section_duration_samples = int(round((beats_per_harmony_section * quarter_duration_in_seconds) * audio.sample_rate))
    start = 0
    harmony_section_starts = []
    while start < len(audio):
        harmony_section_starts.append(start)
        start += harmony_section_duration_samples

    density_section_duration_samples = int(round((beats_per_harmony_section * quarter_duration_in_seconds) * audio.sample_rate))
    start = 0
    density_section_starts = []
    while start < len(audio):
        density_section_starts.append(start)
        start += density_section_duration_samples

    max_start = len(audio) - 500000

    chords = [
        [0, 4, 7],
        [0, 5, 9],
        [0, 4, 7],
        [0, 5, 9],
        [0, 4, 7],
        [0, 5, 9],
        [2, 5, 9, 0],
        [2, 5, 7, 10]
    ]

    lowest = 59
    highest = 62

    density_section_starts = [s for s in density_section_starts if s < max_start]
    len_density_section_starts = len(density_section_starts)
    densities = np.linspace(1, 70, len_density_section_starts)

    halfstep_steps = np.linspace(0, 1, len_density_section_starts, - 10)
    halfstep_steps = list(halfstep_steps)
    halfstep_steps += [1.0] * 10

    i = 0
    for density, section_start in zip(densities, density_section_starts):

        coords = [
            (1, 1),
            (3, 1),
            (5, 1),
            (6, 1),
            (7, 1),
        ]
        for x, y in coords:
            chords[x][y] = round(4.0 + (1 - halfstep_steps[i]), 2)

        coords = [
            (0, 1),
            (2, 1),
            (4, 1),
        ]
        for x, y in coords:
            chords[x][y] = round(5.0 - (1 - halfstep_steps[i]), 2)

        i += 1

        for _ in range(int(density)):
            start = np.random.randint(section_start, section_start + density_section_duration_samples)
            for tick, next_tick in zip(harmony_section_starts[:-1], harmony_section_starts[1:]):
                if tick <= start < next_tick:
                    break

            chord_index = harmony_section_starts.index(tick) % len(chords)
            chord_type = chords[chord_index]
            pitch = random.choice([p for p in np.linspace(lowest, highest, (highest - lowest) * 100, endpoint=False) if round(p % 12, 2) in chord_type])
            note = marimba.get_note(pitch)

            audio.add(start, note)

        if random.random() < .3:
            lowest = max([36, lowest - 1])
        if random.random() < .6:
            highest = min([95, highest + 1])

    audio.write_wav(func_name)
    print 'Done running {}.'.format(func_name)


def dissonant_counterpoint(items, n=100):
    len_items = len(items)
    indexes = np.arange(len_items)
    weights = np.ones(len_items)

    choices = []
    for _ in range(n):
        # Make weights sum to 1.0
        weights /= weights.sum()

        # Weighted choice
        index = np.random.choice(indexes, p=weights)
        choices.append(items[index])

        # Set the weight of the option that was picked to half of the lowest weighted item
        weights[index] = min(weights) / 2

        # Double the weight of any option that wasn't picked
        weights = weights * 2

    return choices


def dissonant_counterpoint_experiment():
    func_name = 'dissonant_counterpoint_experiment'
    print 'Running {}...'.format(func_name)

    marimba = Marimba()

    n_beats = 120
    quarter_duration_in_seconds = .5
    audio_duration = int(n_beats * quarter_duration_in_seconds) + 3
    audio = Audio(audio_duration)
    beat_duration = int(quarter_duration_in_seconds * audio.sample_rate)

    pitch_classes = dissonant_counterpoint(range(12), n=n_beats)

    start = 0
    for i, pc in enumerate(pitch_classes):

        if np.random.random() < .8:
            new_note_duration = np.random.choice(np.linspace(3000, 50000, 100))
            note = marimba.get_staccato_note(pc + 60, new_note_duration=new_note_duration)
        else:
            note = marimba.get_note(pc + 60)

        audio.add(
            start,
            note,
            amplify=np.random.choice(np.linspace(.2, 1.5, 10)),
            pan=np.random.random())
        start += beat_duration

    audio.write_wav(func_name)
    print 'Done running {}.'.format(func_name)


def ratio_to_rounded_pitch_class(m, n):
    return round((ratio_to_cents(m, n, round_decimal_places=None) % 1200) / 100, 2)


def harmonic_to_rounded_pitch_class(h):
    return ratio_to_rounded_pitch_class(h, 1)


def get_odd_harmonics_scale(root=0):
    odd_harmonics = [(x * 2) + 1 for x in range(12)]
    scale = [harmonic_to_rounded_pitch_class(h) for h in odd_harmonics]

    zipped = zip(odd_harmonics, scale)
    zipped.sort(key=lambda x: x[1])

    harmonics, scale = zip(*zipped)

    scale = [round((pc + root) % 12, 2) for pc in scale]

    return scale, harmonics


def diaphonic_trio_piano_two_sections():
    func_name = 'dissonant_counterpoint_experiment'
    print 'Running {}...'.format(func_name)

    marimba = Marimba()
    audio_duration_seconds = 120
    audio = Audio(audio_duration_seconds)

    odd_harmonics_scale, harmonics = get_odd_harmonics_scale(root=0)
    odd_harmonics_scale_pitch_options = [round(p, 2) for p in np.linspace(36, 96, (96 - 36) * 100, endpoint=False) if round(p % 12, 2) in odd_harmonics_scale]

    harmonics = [round((ratio_to_cents(h, 1, round_decimal_places=None) + 3600) / 100, 2) for h in range(1, 25)]

    max_start = len(audio) - 100000

    start_options = [int(s) for s in np.linspace(0, int(max_start / 2), audio_duration_seconds * 3)]
    for _ in range(len(start_options) * 2):
        start = np.random.choice(start_options)
        midi_number = np.random.choice(odd_harmonics_scale_pitch_options)
        note = marimba.get_note(midi_number)
        audio.add(start, note, pan=np.random.random(), amplify=np.random.choice(np.linspace(.3, 1.3)))

    start_options = [int(s) for s in np.linspace(int(max_start / 2), max_start, audio_duration_seconds * 4)]
    for _ in range(len(start_options) * 3):
        start = np.random.choice(start_options)
        midi_number = np.random.choice(harmonics)
        note = marimba.get_note(midi_number)
        audio.add(start, note, pan=np.random.random(), amplify=np.random.choice(np.linspace(.3, 1.3)))

    audio.write_wav(func_name)
    print 'Done running {}.'.format(func_name)


def different_sections():
    func_name = 'different_sections'
    print 'Running {}...'.format(func_name)


    from diaphonic_trio import Sections

    marimba = Marimba()
    audio_duration_seconds = 120
    audio = Audio(audio_duration_seconds)

    sections = Sections(audio, 32)

    scale_type = [0, 4, 7]
    root = 0
    for section in sections:
        section.root = root
        scale = [(pc + root) % 12 for pc in scale_type]
        section.scale = [p for p in range(60, 73) if p % 12 in scale]
        root = (root + 1) % 12

    max_start = len(audio) - 80000

    n_notes = audio_duration_seconds * 5
    for _ in range(n_notes):
        start = np.random.randint(0, max_start)

        scale = sections.get_by_sample_offset(start).scale

        midi_number = np.random.choice(scale)
        note = marimba.get_note(midi_number)
        audio.add(start, note)

    audio.write_wav(func_name)
    print 'Done running {}.'.format(func_name)


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
    # multiple_random_tempos()
    # sections()
    # overlapping_sections()
    # overlapping_sections_with_weird_modulation()
    # dissonant_counterpoint_experiment()
    # diaphonic_trio_piano_two_sections()
    different_sections()
