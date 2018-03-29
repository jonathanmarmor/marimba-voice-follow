import os
from datetime import datetime

import numpy as np
import librosa


class Audio(object):
    def __init__(self,
            duration_seconds=None,
            duration_samples=None,
            initial_audio=None,
            initial_audio_filename=None,
            initial_audio_mono=False,
            mono=False,
            sample_rate=44100):

        self.duration_seconds = duration_seconds
        self.duration_samples = duration_samples
        self.initial_audio = initial_audio
        self.initial_audio_filename = initial_audio_filename
        self.initial_audio_mono = initial_audio_mono
        self.mono = mono
        self.sample_rate = sample_rate

        if initial_audio_filename:
            initial_audio, _ = librosa.load(
                    initial_audio_filename,
                    mono=initial_audio_mono,
                    sr=sr)

        if initial_audio is not None:
            self.n_samples = len(initial_audio)
            self._audio = self._make_audio_array(self.n_samples)
            self._audio += initial_audio

        elif duration_seconds:
            self.n_samples = int(sample_rate * duration_seconds)
            self._audio = self._make_audio_array(self.n_samples)

        elif duration_samples:
            self.n_samples = duration_samples
            self.duration_seconds = self.n_samples / sample_rate
            self._audio = self._make_audio_array(self.n_samples)

    def _make_audio_array(self, n_samples):
        if self.mono:
            return np.zeros(n_samples, dtype=np.float32)
        else:
            return np.zeros([2, n_samples], dtype=np.float32)

    def add(self, start, clip, clip_right=None, pan=0.5, amplify=1.0):
        len_clip = len(clip)

        if self.mono:
            self._audio[start:start + len_clip] += clip
        else:
            if clip_right is None:
                clip_right = clip

            len_clip_right = len(clip_right)

            self._audio[0, start:start + len_clip] += clip * (1 - pan) * amplify
            self._audio[1, start:start + len_clip_right] += clip_right * pan * amplify


    def __len__(self):
        return self._audio.shape[-1]

    def write_wav(self, prefix, output_parent_dir='output'):
        output_dir = os.path.join(output_parent_dir, prefix)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = '{}-{}.wav'.format(prefix, timestamp)
        filename = os.path.join(output_dir, filename)

        librosa.output.write_wav(filename, self._audio, sr=self.sample_rate)