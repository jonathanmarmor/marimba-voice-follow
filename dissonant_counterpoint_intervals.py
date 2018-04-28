"""See https://math.dartmouth.edu/~ahb/papers/dc.pdf"""

from collections import Counter

import numpy as np



def dissonant_counterpoint_intervals():
    """Dissonant Counterpoint based on melodic intervals rather than pitchclasses"""

    pitch_classes = range(12)
    weights = np.ones(12)

    interval_types = range(7)
    interval_types_weights = np.ones(7)
    pc_to_interval_type_map = [i if i <= 6 else 12 - i for i in range(12)]

    previous = np.random.choice(pitch_classes)
    previous_interval_type = np.random.choice(interval_types)

    while True:

        # sdlkfmaslkdmalksdfm=

        # Make weights sum to 1.0
        weights /= weights.sum()

        # Weighted choice
        pitch_class = np.random.choice(pitch_classes, p=weights)

        # Set the weight of the option that was picked to half of the lowest weighted item
        weights[pitch_class] = min(weights) / 2

        # Double the weight of any option that wasn't picked
        weights *= 2

        yield pitch_class














# def diss_count_intervals(n=50, starting_pitchclass=None):
#     """Dissonant Counterpoint of melodic intervals, not pitch classes"""

#     if starting_pitchclass == None:
#         starting_pitchclass = np.random.randint(12)
#     pc = starting_pitchclass

#     intervals = range(7)
#     weights = np.ones(7)

#     result = []
#     for _ in range(n):

#         # Make weights sum to 1.0
#         weights /= weights.sum()

#         # Weighted choice
#         index = np.random.choice(indexes, p=weights)

#         # Set the weight of the option that was picked to half of the lowest weighted item
#         weights[index] = min(weights) / 2

#         # Double the weight of any option that wasn't picked
#         weights *= 2








# def diss_count_pcs_and_intervals():
#     """Create the pattern of dissonant counterpoint both in the absolute pitch
#        classes used and in the intervals between consecutive pitches."""


#     pitchclasses = np.arange(12)
#     pitchclasses_weights = np.ones(12)

#     intervals = np.arange(12)
#     intervals_weights = np.ones(12)

#     while True:

#         pitchclasses_weights /= pitchclasses_weights.sum()
#         intervals_weights /= intervals_weights.sum()


