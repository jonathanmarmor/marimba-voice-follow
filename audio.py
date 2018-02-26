import os
from datetime import datetime

import numpy as np
import librosa


class Audio(object):
    def __init__(self, duration_seconds=None, duration_samples=None, channels=1, sample_rate=44100):
        self.sample_rate = sample_rate

        if duration_seconds:
            self.duration_seconds = duration_seconds
            self.n_samples = int(sample_rate * duration_seconds)
        elif duration_samples:
            self.n_samples = duration_samples
            self.duration_seconds = self.n_samples / sample_rate

        self.channels = channels

        if self.channels == 2:
            self._audio = np.zeros([2, self.n_samples], dtype=np.float32)
        elif self.channels == 1:
            self._audio = np.zeros(self.n_samples, dtype=np.float32)
        else:
            raise Exception('Mono or Stereo only, buddy.')

    def add(self, start, clip, clip_right=None, pan=0.5):
        len_clip = len(clip)

        if self.channels == 1:
            self._audio[start:start + len_clip] += clip

        elif self.channels == 2:
            if clip_right is None:
                clip_right = clip

            len_clip_right = len(clip_right)

            self._audio[0, start:start + len_clip] += clip * pan * .1
            self._audio[1, start:start + len_clip_right] += clip_right * pan * .1

    def len(self):
        return len(self._audio)

    def write_wav(self, prefix, output_parent_dir='output'):
        output_dir = os.path.join(output_parent_dir, prefix)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = '{}-{}.wav'.format(prefix, timestamp)
        filename = os.path.join(output_dir, filename)

        librosa.output.write_wav(filename, self._audio, sr=self.sample_rate)