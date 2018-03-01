import numpy as np


class Music(object):
    '''A MIDI piano roll and synced audio numpy arrays'''

    def __init__(self,
            duration_in_quarters=128,
            quarters_subdivisions=48,
            min_pitch=36.0,
            max_pitch=96.0,
            pitches_per_halfstep=2,
            quarters_per_minute=60,
            sample_rate=44100):

        self.duration_in_quarters = duration_in_quarters
        self.quarters_subdivisions = quarters_subdivisions
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.pitches_per_halfstep = pitches_per_halfstep
        self.quarters_per_minute = quarters_per_minute
        self.sample_rate = sample_rate

        self.n_ticks = self.duration_in_quarters * self.quarters_subdivisions

        ticks_per_minute = self.quarters_per_minute * self.quarters_subdivisions
        self.tick_duration_in_seconds = 60.0 / ticks_per_minute
        self.tick_duration_in_samples = int(round(self.tick_duration_in_seconds * self.sample_rate))

        # Re-set these values after rounding to the nearest integer sample
        self.tick_duration_in_seconds = float(self.tick_duration_in_samples) / self.sample_rate
        self.quarter_duration_in_seconds = self.tick_duration_in_seconds * self.quarters_subdivisions
        # self.quarters_per_minute =

        self.n_samples = self.tick_duration_in_samples * self.n_ticks

        self.audio = np.zeros([2, self.n_samples], dtype=np.float32)


        self.n_halfsteps = max_pitch - min_pitch
        self.n_pitches = int(self.n_halfsteps * pitches_per_halfstep)
        self.midi_notes = np.linspace(min_pitch, max_pitch, self.n_pitches, endpoint=False)

        self.piano_roll = np.zeros((self.n_pitches, self.n_ticks))
