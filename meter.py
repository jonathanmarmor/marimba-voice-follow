"""Usage

"""


import numpy as np

from utils import scale  #, get_beat_starts


def seconds_to_samples(seconds, sample_rate=44100):
    return int(round(seconds * sample_rate))

seconds_to_samples_vectorized = np.vectorize(seconds_to_samples)


def get_beat_starts(beat_duration, total_duration):
    '''Given a tempo and a duration to fill, get the start offsets of all beats'''
    # beat_duration = 60.0 / bpm
    n_beats = int(total_duration // beat_duration)
    effective_duration = n_beats * beat_duration
    starts = np.linspace(0, effective_duration, n_beats, endpoint=False)
    return starts


class Beat(object):
    def __init__(self, start, duration, index, name, sample_rate=44100):
        self.start = start
        self.duration = duration
        self.next_start = start + duration

        self.start_samples = seconds_to_samples(self.start, sample_rate=sample_rate)
        self.duration_samples = seconds_to_samples(self.duration, sample_rate=sample_rate)
        self.next_start_samples = seconds_to_samples(self.next_start, sample_rate=sample_rate)

        self.index = index

        hierarchy = {
            'sixteenth': 16,
            'eighth': 8,
            'quarter': 4,
            'half': 2,
            'whole': 1
        }
        self.position_in_bar = index % hierarchy[name]

        self.

        self.name = name

    def __repr__(self):
        return '<Beat {}, start: {}, duration: {}>'.format(
            self.index,
            self.start,
            self.duration)


class Meter(object):
    def __init__(self, total_duration_seconds, quarter_duration_seconds=None, bpm=60.0, sample_rate=44100):
        if quarter_duration_seconds == None:
            quarter_duration_seconds = 60.0 / bpm
        self.total_duration_seconds = total_duration_seconds
        self.quarter_duration_seconds = quarter_duration_seconds
        self.bpm = bpm
        self.sample_rate = sample_rate


        self.all_layers = []
        self.layers_by_name = {}

        relative_durations = [.25, .5, 1.0, 2.0, 4.0]
        self.names = ['sixteenth', 'eighth', 'quarter', 'half', 'whole']

        for relative_duration, name in zip(relative_durations, self.names):
            beat_duration = self.quarter_duration_seconds * relative_duration
            if name is not 'quarter':
                setattr(self, '{}_duration_seconds'.format(name), beat_duration)

            starts = get_beat_starts(beat_duration, self.total_duration_seconds)
            setattr(self, 'starts_{}_seconds'.format(name), starts)

            beats = []
            setattr(self, '{}s'.format(name), beats)
            self.all_layers.append(beats)
            self.layers_by_name[name] = beats

            for index, start in enumerate(starts):
                beat = Beat(start, beat_duration, index, name, sample_rate=self.sample_rate)
                beats.append(beat)

    def get_by_seconds_offset(self, seconds_offset):
        happening_now = []
        for layer in self.all_layers:
            for beat in layer:
                if beat.start <= seconds_offset < beat.next_start:
                    happening_now.append(beat)
                    break
        return happening_now

    def get_by_samples_offset(self, samples_offset):
        happening_now = []
        for layer in self.all_layers:
            for beat in layer:
                if beat.start_samples <= samples_offset < beat.next_start_samples:
                    happening_now.append(beat)
                    break
        return happening_now

    def get_between_samples(self, a, b):
        result = {}
        for name in self.names:
            result[name] = []

            layer = self.layers_by_name[name]
            for beat in layer:
                if a <= beat.start_samples < b:
                    encountered = True
                    result[name].append(beat)
                if beat.start_samples >= b:
                    # If we've already gone past the window we can stop iterating
                    break

        return result
