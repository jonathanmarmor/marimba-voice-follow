#!/usr/bin/env python

'''Turn a directory containing one or more samples into a numpy array containing pitch-shifted copies of the sample at every cent transposition'''

import os
from glob import glob
from datetime import datetime

import scipy
import librosa
import numpy as np


# TODO: function to analyse a sample to determine what pitch it is
# TODO: function to seek through a track and find candidates for samples, eg with stable pitches and clear starts/stops

def load_samples(directory):
    print '\nLoading samples...'
    samples = {}
    directory = os.path.join(directory, '*.wav')
    for filename in glob(directory):
        y, _ = librosa.load(filename, sr=44100, mono=True)

        # Assume the pitch name is the filename
        pitch = os.path.basename(filename).replace('.wav', '')

        midi_number = librosa.note_to_midi(pitch)
        samples[midi_number] = y

        print '\t', filename
    print 'Done loading samples.\n'
    return samples


def pitch_shift(sample, from_note, to_note):
    ratio = librosa.midi_to_hz(from_note) / librosa.midi_to_hz(to_note)
    n_samples = int(np.ceil(len(sample) * ratio))

    return scipy.signal.resample(sample, n_samples, axis=-1)


def find_closest_midi_number(samples_midi_numbers, midi_number):
    min_diff = float('+inf')
    closest = None
    for sample_midi_number in samples_midi_numbers:
        diff = np.abs(sample_midi_number - midi_number)
        if diff < min_diff:
            min_diff = diff
            closest = sample_midi_number

    return closest


def save_npy(array, filename, output_dir):
    print '\nSaving numpy array to file...'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = '{}-{}'.format(filename, timestamp)
    filename = os.path.join(output_dir, filename)

    # TODO: use np.savez_compressed instead of np.save
    np.save(filename, array)
    print 'Done saving numpy array.\n'


def midi_number_to_index(midi_number, min_midi_number=36.0, max_midi_number=96.0):
    min_midi_number_times_100 = round(min_midi_number * 100)
    return int(round(midi_number * 100) - min_midi_number_times_100)


def make_cents_samples(original_samples, min_midi_number=36.0, max_midi_number=96.0):
    '''Pitch shift samples so there is a sample for every cent transposition over several octaves'''

    print '\nMaking cents samples...'

    length = 50000

    original_samples_keys = original_samples.keys()
    original_samples_keys.sort()

    n_notes_cents = int(round(max_midi_number * 100)) - int(round(min_midi_number * 100))
    midi_numbers_cents = np.linspace(min_midi_number, max_midi_number, n_notes_cents, endpoint=False)

    wavetable = np.zeros([n_notes_cents, length], dtype=np.float32)

    for midi_number in midi_numbers_cents:
        print midi_number,

        closest_midi_number = find_closest_midi_number(original_samples_keys, midi_number)

        if midi_number == closest_midi_number:
            print 'already exists...',
            sample = original_samples[closest_midi_number]
        else:
            print 'shifting pitch from', closest_midi_number, '...',
            sample = original_samples[closest_midi_number][:length]
            sample = pitch_shift(sample, closest_midi_number, midi_number)

        print 'normalizing...',
        sample = librosa.util.normalize(sample[:length])

        print 'fade out...',
        sample[-10000:] *= np.linspace(1, 0, 10000)

        print 'putting in wavetable...',
        index = midi_number_to_index(
                midi_number,
                min_midi_number=min_midi_number,
                max_midi_number=max_midi_number)
        wavetable[index, :min(len(sample), length)] = sample

        print 'done with', midi_number

    print 'Done making cents samples!'
    return wavetable


def make_sample_library(
        samples_directory,
        output_path,
        output_filename,
        min_midi_number=36.0,
        max_midi_number=96.0):

    original_samples = load_samples(samples_directory)
    cents_samples = make_cents_samples(
            original_samples,
            min_midi_number=min_midi_number,
            max_midi_number=max_midi_number)

    save_npy(cents_samples, output_filename, output_path)


def make_marimba_samples():
    min_midi_number = 36.0
    max_midi_number = 96.0
    make_sample_library(
            'samples/marimba/original',
            'samples/marimba',
            'marimba-cents',
            min_midi_number=min_midi_number,
            max_midi_number=max_midi_number)


if __name__ == '__main__':
    make_marimba_samples()
