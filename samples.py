#!/usr/bin/env python

import os
from glob import glob

import scipy
import librosa
import numpy as np

from utils import write_wav


def load_samples():
    print '\nLoading samples...'
    samples = {}
    for filename in glob('samples/*.wav'):
        y, _ = librosa.load(filename, sr=44100, mono=True)
        pitch = os.path.basename(filename).replace('.wav', '')
        midi_number = librosa.note_to_midi(pitch)
        samples[midi_number] = y

        print '\t', filename
    print 'Done loading samples.'
    return samples


def make_wavetable(length=50000, min_note=36, max_note=96):
    samples = load_samples()

    print '\nBuilding wavetable...'
    wavetable = np.zeros([max_note - min_note, length], dtype=np.float32)
    for note in range(min_note, max_note):
        print '\t', note,

        closest_note = find_closest_note(samples, note)

        if note == closest_note:
            sample = samples[note]
            print ''
        else:
            print 'shifting pitch from', closest_note
            sample = pitch_shift(samples[closest_note][:length], closest_note, note)

        index = note - min_note
        sample = librosa.util.normalize(sample[:length])
        sample[-10000:] *= np.linspace(1, 0, 10000)
        wavetable[index, :min(len(sample), length)] = sample
    print 'Done building wavetable.'
    return wavetable


def pitch_shift(sample, from_note, to_note):
    sample = sample

    ratio = (librosa.midi_to_hz(from_note) / librosa.midi_to_hz(to_note))[0]
    n_samples = int(np.ceil(len(sample) * ratio))
    # print from_note, to_note, ratio, n_samples
    return scipy.signal.resample(sample, n_samples, axis=-1)


def find_closest_note(samples, note):
    min_diff = float('+inf')
    closest = None
    for n in samples:
        diff = np.abs(n - note)
        if diff < min_diff:
            min_diff = diff
            closest = n

    return closest


def test_samples(wavetable):
    audio = np.zeros(50000 * len(wavetable), dtype=np.float32)
    for i in range(len(wavetable)):
        w = wavetable[i]
        audio[i * 50000:i * 50000 + len(w)] = w

    write_wav(audio, 'samples-test')


def random_notes(wavetable):
    """Chose a random note every 6250 samples"""
    audio = np.zeros(50000 * len(wavetable), dtype=np.float32)

    note_indexes = range(len(wavetable))
    tick = 0
    while tick < len(audio) - 50000:
        note_index = np.random.choice(note_indexes)
        note = wavetable[note_index]

        audio[tick:tick + len(note)] += note

        tick += 6250

    write_wav(audio, 'samples-test')


def random_notes_2(wavetable):
    """Choose 100 random notes and put them in random locations throughout the duration of the audio containing the original samples"""
    audio = np.zeros(50000 * len(wavetable), dtype=np.float32)

    len_audio = len(audio) - 50000
    all_ticks = range(len_audio)

    note_indexes = range(len(wavetable))

    n_notes = 100

    for _ in range(n_notes):
        note_index = np.random.choice(note_indexes)
        note = wavetable[note_index]

        start = np.random.choice(all_ticks)

        audio[start:start + len(note)] += note

    write_wav(audio, 'samples-test')


def microtonal_experiment_1(wavetable):
    print 'Running microtonal_experiment_1...'
    audio = np.zeros(44100 * 30, dtype=np.float32)

    # the marimba recordings have a duration of 50000 samples.
    # Don't start a recording within 50000 samples of the end of the audio
    len_audio = len(audio) - 50000

    all_ticks = range(len_audio)
    note_indexes = range(len(wavetable))
    cents = np.linspace(-.5, .49, 100)
    n_notes = 100
    for i in range(n_notes):
        print '\t', i, 'of', n_notes
        note_index = np.random.choice(note_indexes)
        note = wavetable[note_index]
        cents_adjustment = np.random.choice(cents)
        note = pitch_shift(note, note_index, note_index + cents_adjustment)
        start = np.random.choice(all_ticks)
        audio[start:start + len(note)] += note
    write_wav(audio, 'samples-test')
    print 'Done running microtonal_experiment_1.'


if __name__ == '__main__':
    wavetable = make_wavetable()
    # test_samples(wavetable)
    # random_notes_2(wavetable)
    microtonal_experiment_1(wavetable)
