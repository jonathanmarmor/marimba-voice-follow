import numpy as np

from utils import scale


class Section(object):
    def __init__(self, start, next_start, index, of_n_sections, sections):
        self.start = start
        self.next_start = next_start
        self.end = next_start - 1
        self.duration = self.end - self.start

        self.index = index
        self.of_n_sections = of_n_sections

        self.parent = sections

    def __repr__(self):
        return '<Section {} of {}, start: {}, duration: {}>'.format(
            self.index,
            self.of_n_sections,
            self.start,
            self.duration)


class Sections(list):
    def __init__(self, audio, n_sections=None, relative_durations=None):
        self.audio = audio

        len_audio = len(audio)

        if n_sections:
            self.n_sections = n_sections
            self.starts = [int(round(start)) for start in np.linspace(0, len_audio, n_sections, endpoint=False)]

        elif relative_durations:
            self.n_sections = len(relative_durations)
            sum_relative_durations = sum(relative_durations)

            durations = [scale(d, 0, sum_relative_durations, 0, len_audio) for d in relative_durations]
            start = 0
            self.starts = []
            for d in durations:
                self.starts.append(start)
                start += d
            self.starts = [int(round(s)) for s in self.starts]

        self.next_starts = self.starts[1:] + [len_audio + 1]

        index = 0
        for start, next_start in zip(self.starts, self.next_starts):
            section = Section(start, next_start, index, self.n_sections, self)
            self.append(section)
            index += 1

    def get_by_sample_offset(self, sample_offset):
        for section in self:
            if section.start <= sample_offset < section.next_start:
                return section


def sections_usage():
    from marimba_samples import Marimba
    from audio import Audio

    marimba = Marimba()
    audio_duration_seconds = 120
    audio = Audio(audio_duration_seconds)

    layer_1 = Sections(audio, 13)
    layer_2 = Sections(audio, 17)

    offsets = [int(round(o)) for o in np.linspace(0, len(audio), 20, endpoint=False)]
    for offset in offsets:
        layer_1_index = layer_1.get_by_sample_offset(offset).index
        layer_2_index = layer_2.get_by_sample_offset(offset).index
        print offset, layer_1_index, layer_2_index

    # print sections[3].duration

