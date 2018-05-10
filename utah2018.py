#!/usr/bin/env python

import numpy as np

from marimba_samples import Marimba
from audio import Audio
from meter import Meter
from sections import Sections


class Utah2018(object):
    def __init__(self, version):
        self.version = version

        self.duration_seconds = 200
        self.setup()
        self.planning()
        self.go()
        self.closeout()

    def setup(self):
        self.name = 'Utah2018-{}'.format(self.version)
        print '\nRunning {}...'.format(self.name)
        self.marimba = Marimba()
        self.audio = Audio(self.duration_seconds + 5)
        self.len_audio = len(self.audio) - (44100 * 5)

    def closeout(self):
        self.audio.write_wav(self.name)
        print 'Done running {}.\n'.format(self.name)

    def planning(self):
        self.meter = Meter(self.duration_seconds, bpm=120)

        self.registers = [
            range(68, 77),
            range(77, 87),
            range(54, 68),
            range(87, 96),
            range(36, 48),
            range(77, 87),
            range(68, 77),
            range(87, 96),
        ]

        self.layer_density_order = [
            'eighth',
            'half',
            'quarter',
            'sixteenth',
            'half',
            'whole',
            'quarter',
            'sixteenth'
        ]

        self.n_layers_sections = Sections(8, self.len_audio)
        n = 1
        for section in self.n_layers_sections:
            section.n_layers = n
            n += 1

        harmonic_rhythm_duration = 2.0

        self.harmony_sections = Sections(int(self.duration_seconds / harmonic_rhythm_duration), self.len_audio)
        for harmony_section_index, harmony in enumerate(self.harmony_sections):
            harmony_index = harmony.index % 16

            if harmony_index in [0, 1, 2, 3]:
                harmony.harmony = [0, 4, 7]
                harmony.scale = [0, 2, 4, 5, 7, 9, 11]

            elif harmony_index in [4, 5, 6, 7]:
                harmony.harmony = [2, 5, 7, 11]
                harmony.scale = [0, 2, 4, 5, 7, 9, 11]

            elif harmony_index in [8, 9]:
                harmony.harmony = [0, 4, 7]
                harmony.scale = [0, 2, 4, 5, 7, 9, 11]

            elif harmony_index in [10]:
                harmony.harmony = [0, 2, 5, 9]
                harmony.scale = [0, 2, 4, 5, 7, 9, 11]

            elif harmony_index in [11]:
                harmony.harmony = [2, 5, 7, 11]
                harmony.scale = [0, 2, 4, 5, 7, 9, 11]

            elif harmony_index in [12, 13, 14]:
                harmony.harmony = [0, 4, 7]
                harmony.scale = [0, 2, 4, 5, 7, 9, 11]

            elif harmony_index in [15]:
                harmony.harmony = [2, 5, 7, 11]
                harmony.scale = [0, 2, 4, 5, 7, 9, 11]

    def go(self):
        i = 0
        for section in self.n_layers_sections:
            beats = self.meter.get_between_samples(section.start, section.next_start)

            for layer in range(section.n_layers):
                duration_name = self.layer_density_order[layer]
                for beat in beats[duration_name]:
                    if np.random.random() > .22:
                        harmony = self.harmony_sections.get_by_sample_offset(beat.start_samples)

                        # print i
                        i += 1

                        beat_index = beat.index % 8
                        if beat_index == 0:
                            amplify = np.random.choice(np.linspace(.7, 1.1, 10))
                            pitch_options = [p for p in self.registers[layer] if p % 12 in harmony.harmony]
                            pitch = np.random.choice(pitch_options)
                            note = self.marimba.get_note(pitch)

                        elif beat_index == 4:
                            amplify = np.random.choice(np.linspace(.5, .8, 10))
                            pitch_options = [p for p in self.registers[layer] if p % 12 in harmony.harmony]
                            pitch = np.random.choice(pitch_options)
                            note = self.marimba.get_note(pitch)

                        elif beat_index in [2, 6]:
                            amplify = np.random.choice(np.linspace(.3, .6, 10))
                            pitch_options = [p for p in self.registers[layer] if p % 12 in harmony.scale]
                            pitch = np.random.choice(pitch_options)
                            note = self.marimba.get_note(pitch)

                        elif beat_index in [1, 3, 5, 7]:
                            amplify = np.random.choice(np.linspace(.1, .4, 10))
                            pitch_options = [p for p in self.registers[layer] if p % 12 in harmony.scale]
                            pitch = np.random.choice(pitch_options)
                            note = self.marimba.get_note(pitch)

                        self.audio.add(
                            beat.start_samples,
                            note,
                            amplify=amplify,
                            pan=np.random.choice(np.linspace(.15, .85, 10)))


if __name__ == '__main__':
    Utah2018('0.0.1')
