#!/usr/bin/env python

import librosa
import numpy as np

from marimba_samples import Marimba
# from audio import Audio
from utils import write_wav


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


def make_music():
    sr = 44100
    denis_audio, _ = librosa.load('denis-curran-short.mp3', mono=True, sr=sr)
    cqt = np.abs(librosa.cqt(denis_audio, sr=sr)).T
    denis_pitches = np.argmax(cqt, 1) * (np.max(cqt, 1) > .7)

    # audio = Audio(len(denis_audio) / sr, channels=2)  # add 3 seconds at the end so we don't have to cut off the last marimba note
    # audio.add(0, denis_audio)
    audio = np.zeros([len(denis_audio), 2])
    audio = (audio.T + denis_audio).T

    marimba = Marimba()

    previous_notes = np.ones(marimba.n_notes) * -100
    previous_note_i = -100
    chord = random_chord()

    hop_length = 512

    for i, p in enumerate(denis_pitches):
        print p
        if p > 0 and i - previous_note_i > 3:

            if i - previous_note_i > 50:
                chord = random_chord()

            t = i * hop_length

            p = note_in_chord(chord, p)
            while p >= marimba.n_notes:
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

            note = marimba.get_note(p)
            length = min(len(audio) - t, len(note))

            pan = np.random.random()

            # audio.add(t, note, pan=pan)
            audio[t:t + length, 0] += note[:length] * pan * .1
            audio[t:t + length, 1] += note[:length] * (1 - pan) * .1

            previous_note_i = i
            previous_notes[p] = i

    # audio.write_wav('denis-marimba')
    write_wav(audio, 'denis-marimba')


if __name__ == '__main__':
    make_music()
