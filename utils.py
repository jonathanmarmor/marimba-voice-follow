import os
from datetime import datetime

import librosa


def write_wav(audio, prefix, output_parent_dir='output'):
    output_dir = os.path.join(output_parent_dir, prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = '{}-{}.wav'.format(prefix, timestamp)
    filename = os.path.join(output_dir, filename)

    librosa.output.write_wav(filename, audio, sr=44100)
