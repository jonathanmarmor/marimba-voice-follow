#!/usr/bin/env python

import math
import random

import numpy as np

from marimba_samples import Marimba
from audio import Audio
from utils import random_from_range, ratio_to_cents




# def ratio_to_rounded_pitch_class(m, n):
#     return round((ratio_to_cents(m, n, round_decimal_places=None) % 1200) / 100, 2)


# def harmonic_to_rounded_pitch_class(h):
#     return ratio_to_rounded_pitch_class(h, 1)


# def get_odd_harmonics_scale(root=0):
#     odd_harmonics = [(x * 2) + 1 for x in range(12)]
#     scale = [harmonic_to_rounded_pitch_class(h) for h in odd_harmonics]

#     zipped = zip(odd_harmonics, scale)
#     zipped.sort(key=lambda x: x[1])

#     harmonics, scale = zip(*zipped)

#     scale = [round((pc + root) % 12, 2) for pc in scale]

#     return scale, harmonics


# odd_harmonics_scale, harmonics = get_odd_harmonics_scale(root=0)
# odd_harmonics_scale_pitch_options = [round(p, 2) for p in np.linspace(36, 96, (96 - 36) * 100, endpoint=False) if round(p % 12, 2) in odd_harmonics_scale]

# harmonics = [round((ratio_to_cents(h, 1, round_decimal_places=None) + 3600) / 100, 2) for h in range(1, 25)]

# #######


# harmonics = np.array([1, 17, 9, 19, 5, 21, 11, 23, 3, 13, 7, 15])

# array = []
# for multiplier in [16, 8, 4, 2, 1]:
#     row = harmonics * multiplier
#     array.append(row)
# array = np.array(array)

# flattened = sorted(list(array.flatten()), reverse=True)




class Section(object):
    def __init__(self, start, next_start, index, of_n_sections, sections):
        self.start = start
        self.next_start = next_start
        self.end = next_start - 1
        self.duration = self.end - self.start

        self.index = index
        self.of_n_sections = of_n_sections

    def __repr__(self):
        return '<Section {} of {}, start: {}, duration: {}>'.format(
            self.index,
            self.of_n_sections,
            self.start,
            self.duration)


class Sections(list):
    def __init__(self, audio, n_sections):
        self.audio = audio
        self.n_sections = n_sections

        self.starts = [int(round(start)) for start in np.linspace(0, len(audio), n_sections, endpoint=False)]

        nodes = self.starts + [len(audio) + 1]
        index = 0
        for start, next_start in zip(nodes[:-1], nodes[1:]):
            section = Section(start, next_start, index, n_sections, self)
            self.append(section)
            index += 1

    def get_by_sample_offset(self, sample_offset):
        for section in self:
            if section.start <= sample_offset < section.next_start:
                return section


def sections_usage():
    marimba = Marimba()
    audio_duration_seconds = 120
    audio = Audio(audio_duration_seconds)

    sections = Sections(audio, 16)

    sections.get_by_sample_offset(np.random.choice(np.linspace(0, len(audio), 1000))).index

    sections[3].duration




# class PianoHarmonies(object):
#     def __init__(self, audio):
#         self.min_midi_pitch = 36
#         self.max_midi_pitch = 96

#         self.harmony_sequence = self.build_harmony_sequence()

#         self.sections = Sections(audio, len(self.harmony_sequence))

#     def build_harmony_sequence(self):
#         harmony_sequence = []


#         return harmony_sequence


#     def get_at_sample_offset(self, sample_offset):


#         return pitches



def main():
    marimba = Marimba()
    audio_duration_seconds = 120
    audio = Audio(audio_duration_seconds)

    piano_harmonies = PianoHarmonies(audio)

    max_start = len(audio) - 80000

    n_notes = audio_duration_seconds * 3
    for _ in range(n_notes):
        start = np.random.randint(0, max_start)
        scale = piano_harmonies.get_at_sample_offset(start)
        midi_number = np.random.choice(scale)
        note = marimba.get_note(midi_number)
        audio.add(start, note)


if __name__ == '__main__':
    main()
