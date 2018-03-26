import os
import math
from datetime import datetime

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
