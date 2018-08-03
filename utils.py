import os
import math
from datetime import datetime
import itertools
import collections

import numpy as np
import librosa


def write_wav(audio, prefix, output_parent_dir='output'):
    output_dir = os.path.join(output_parent_dir, prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = '{}-{}.wav'.format(prefix, timestamp)
    filename = os.path.join(output_dir, filename)

    librosa.output.write_wav(filename, audio, sr=44100)


def random_from_range(a, b, size=None):
    return (b - a) * np.random.random(size=size) + a


def ratio_to_cents(m, n, round_decimal_places=2):
    m = float(m)
    cents = 1200 * math.log(m / n) / math.log(2)
    if round_decimal_places is not None:
        cents = round(cents, round_decimal_places)
    return cents


def scale(x, original_low, original_high, target_low=0.0, target_high=1.0):
    """Project `x`'s position within `original_low` and `original_high` to the same position within `target_low` and `target_high`"""
    return ((target_high - target_low) * (float(x) - original_low)) / (original_high - original_low) + target_low


def seconds_to_samples(seconds, sample_rate=44100):
    return int(round(seconds * sample_rate))

seconds_to_samples_vectorized = np.vectorize(seconds_to_samples)


def get_beat_starts(bpm, total_duration, sample_rate=44100):
    '''Given a tempo and a duration to fill, get the start offsets of all beats, in terms of samples'''
    beat_duration = 60.0 / bpm
    n_beats = int(total_duration // beat_duration)
    effective_duration = n_beats * beat_duration
    starts_seconds = np.linspace(0, effective_duration, n_beats, endpoint=False)
    starts_samples = seconds_to_samples_vectorized(starts_seconds)
    return starts_samples


def pairwise(iterable):
    """
    >>> list(pairwise(range(5)))
    [(0, 1), (1, 2), (2, 3), (3, 4)]

    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


def find_loops(seq):
    seen = []
    dupes = collections.defaultdict(list)
    for i, item in enumerate(seq):
        dupes[item].append(i)

    loops = collections.defaultdict(list)
    for item in dupes:
        indexes = dupes[item]
        if len(indexes) > 1:
            for a, b in pairwise(indexes):
                loops[seq[a]].append(seq[a:b])

    return loops


def n_wise(iterable, n):
    """Iterate through `iterable` in groups of `n` items.

    >>> list(n_wise(range(5), 2))
    [(0, 1), (1, 2), (2, 3), (3, 4)]

    >>> list(n_wise(range(5), 3))
    [(0, 1, 2), (1, 2, 3), (2, 3, 4)]

    >>> list(n_wise(range(5), 4))
    [(0, 1, 2, 3), (1, 2, 3, 4)]

    """
    iterators = itertools.tee(iterable, n)
    for index, iterator in enumerate(iterators):
        for _ in range(index):
            next(iterator, None)
    return itertools.izip(*iterators)