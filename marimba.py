#!/usr/bin/env python

import os
from glob import glob
import datetime

import scipy
import librosa
import numpy as np


def load_denis():
    denis, _ = librosa.load('denis-curran-short.mp3', mono=True, sr=44100)
    return denis


def get_cqt(denis):
    return np.abs(librosa.cqt(denis, sr=44100)).T


DOM7 = np.array([0, 3, 7, 10])
MIN7 = np.array([0, 3, 7, 10])
MAJ7 = np.array([0, 4, 7, 11])
MAJ = np.array([0, 4, 7])
MIN = np.array([0, 3, 7])
SUS_A = np.array([0, 2, 5, 7])
SUS_B = np.array([0, 2, 4, 7])
SUS_C = np.array([0, 3, 5, 7])
CHORD_TYPES = [DOM7, MIN7, MAJ7, MAJ,  MIN,  SUS_A, SUS_B, SUS_C]
CHORD_TYPE_WEIGHTS = [.3, .25, .2, .075, .075, 0.1/3, 0.1/3, 0.1/3]


def random_chord():
    chord_type_index = np.random.choice(len(CHORD_TYPES), p=CHORD_TYPE_WEIGHTS)
    chord_type = CHORD_TYPES[chord_type_index]
    root = np.random.randint(12)
    chord = (chord_type + root) % 12
    mask = np.zeros(12)
    mask[chord] = 1
    return mask


def note_in_chord(mask, p):
    if mask[p % 12] == 1:
        return p

    min_diff = float('+inf')
    closest = None
    for i in range(len(mask)):
        if mask[i] == 0:
            continue
        diff = min(np.abs((i - p) % 12), np.abs((p - i) % 12))
        if diff < min_diff:
            min_diff = diff
            closest = i

    return closest + (p // 12) * 12


def vocal_notes(denis, cqt, wavetable, hop_length=512):
    pitches = np.argmax(cqt, 1) * (np.max(cqt, 1) > .7)
    audio = np.zeros([len(denis), 2])
    audio = (audio.T + denis).T

    previous_notes = np.ones(len(wavetable)) * -100
    previous_note_i = -100
    chord = random_chord()
    for i in range(len(pitches)):
        p = pitches[i]
        if p > 0 and i - previous_note_i > 3:

            if i - previous_note_i > 50:
                chord = random_chord()

            t = i * hop_length

            p = note_in_chord(chord, p)
            while p >= len(wavetable):
                p -= 12

            if np.random.random() > .5:
                while p > 0 and i - previous_notes[p] < 10:
                    p -= 12
                if p < 0:
                    continue
            else:
                while p < 60 and i - previous_notes[p] < 10:
                    p += 12
                if p >= 60:
                    continue

            note = wavetable[p]
            length = min(len(audio) - t, len(note))

            pan = np.random.random()
            audio[t:t + length, 0] += note[:length] * pan * .1
            audio[t:t + length, 1] += note[:length] * (1 - pan) * .1

            previous_note_i = i
            previous_notes[p] = i

    return audio


def load_samples():
    samples = {}
    for filename in glob('samples/*.wav'):
        y, _ = librosa.load(filename, sr=44100, mono=True)
        pitch = os.path.basename(filename).replace('.wav', '')
        hz = librosa.note_to_midi(pitch)
        samples[hz] = y

        print filename, np.mean(y)

    return samples


def make_wavetable(samples, length=50000, min_note=36, max_note=96):
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


def test_samples(wavetable):
    audio = np.zeros(50000 * len(wavetable), dtype=np.float32)
    for i in range(len(wavetable)):
        w = wavetable[i]
        audio[i * 50000:i * 50000 + len(w)] = w

    librosa.output.write_wav('test.wav', audio, sr=44100)


def make_music(wavetable):
    denis = load_denis()
    cqt = get_cqt(denis)

    audio = vocal_notes(denis, cqt, wavetable)

    if not os.path.exists('output'):
        os.mkdir('output')

    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = 'output/denis-marimba-{}.wav'.format(timestamp)

    librosa.output.write_wav(filename, librosa.util.normalize(audio), sr=44100)


def main():
    samples = load_samples()
    wavetable = make_wavetable(samples)

    # test_samples(wavetable)

    make_music(wavetable)


if __name__ == '__main__':
    main()
