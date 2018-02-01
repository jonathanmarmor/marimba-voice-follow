#!/usr/bin/env python

import os
from glob import glob
import datetime

import scipy
import librosa
import numpy as np


def load_samples():
    samples = {}
    for filename in glob('samples/*.wav'):
        y, _ = librosa.load(filename, sr=44100, mono=True)
        pitch = os.path.basename(filename).replace('.wav', '')
        hz = librosa.note_to_midi(pitch)
        samples[hz] = y

        # print filename, np.mean(y)

    return samples


def make_wavetable(length=50000, min_note=36, max_note=96):
    samples = load_samples()

    wavetable = np.zeros([max_note - min_note, length], dtype=np.float32)
    for note in range(min_note, max_note):

        closest_note = find_closest_note(samples, note)

        if note == closest_note:
            sample = samples[note]
        else:
            sample = pitch_shift(samples[closest_note][:length], closest_note, note)

        index = note - min_note
        sample = librosa.util.normalize(sample[:length])
        sample[-10000:] *= np.linspace(1, 0, 10000)
        wavetable[index, :min(len(sample), length)] = sample

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


def write_wav(audio, output_parent_dir='output', prefix='samples-test'):
    output_dir = os.path.join(output_parent_dir, prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = '{}-{}.wav'.format(prefix, timestamp)
    filename = os.path.join(output_dir, filename)

    librosa.output.write_wav(filename, audio, sr=44100)


def test_samples(wavetable):
    audio = np.zeros(50000 * len(wavetable), dtype=np.float32)
    for i in range(len(wavetable)):
        w = wavetable[i]
        audio[i * 50000:i * 50000 + len(w)] = w

    write_wav(audio)


if __name__ == '__main__':
    wavetable = make_wavetable()
    test_samples(wavetable)
