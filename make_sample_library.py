#!/usr/bin/env python

'''Turn a directory containing one or more samples into a numpy array containing pitch-shifted copies of the sample at every cent transposition'''

import os
from glob import glob
from datetime import datetime

import scipy
import librosa
import numpy as np


# TODO: function to analyse a sample determine what pitch it is
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
    print 'Done loading samples.'
    return samples


def pitch_shift(sample, from_note, to_note):
    # sample = sample  # Is there a reason Andreas did this?

    ratio = (librosa.midi_to_hz(from_note) / librosa.midi_to_hz(to_note))[0]
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = '{}-{}'.format(filename, timestamp)
    filename = os.path.join(output_dir, filename)

    np.save(filename, array)


def make_cents_samples(original_samples, min_midi_number=3600, max_midi_number=9600):
    '''Pitch shift samples so there is a sample for every cent transposition over several octaves'''
    length = 50000

    wavetable = np.zeros([max_midi_number - min_midi_number, length], dtype=np.float32)
    for midi_number in range(min_midi_number, max_midi_number):
        print midi_number,

        closest_midi_number = find_closest_midi_number(samples, midi_number)

        if midi_number == closest_midi_number:
            print 'already exists...',
            sample = samples[midi_number]
        else:
            print 'shifting pitch from', closest_midi_number, '...',

            from_note = closest_midi_number / 100.0
            to_note = midi_number / 100.0

            sample = samples[closest_midi_number][:length]
            sample = pitch_shift(sample, from_note, to_note)

        print 'normalizing...',
        sample = librosa.util.normalize(sample[:length])

        print 'fade out...',
        sample[-10000:] *= np.linspace(1, 0, 10000)

        print 'putting in wavetable...',
        index = midi_number - min_midi_number
        wavetable[index, :min(len(sample), length)] = sample

        print 'done with', midi_number

    print 'Done!'
    return wavetable


def make_sample_library(samples_directory, output_path, output_filename,
        min_midi_number=3600, max_midi_number=9600):

    original_samples = load_samples(samples_directory)
    cents_samples = make_cents_samples(original_samples,
            min_midi_number=min_midi_number, max_midi_number=max_midi_number)
    save_npy(cents_samples, output_filename, output_path)


def make_marimba_samples():
    make_sample_library('samples/marimba/original', 'samples/marimba',
            'marimba-cents', min_midi_number=3600, max_midi_number=9600)


if __name__ == '__main__':
    make_marimba_samples()
