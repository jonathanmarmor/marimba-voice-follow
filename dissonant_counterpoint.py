"""See https://math.dartmouth.edu/~ahb/papers/dc.pdf"""

from collections import Counter

import numpy as np


def choose_and_reweight(indexes, weights):
    # Make weights sum to 1.0
    weights /= weights.sum()

    # Weighted choice
    index = np.random.choice(indexes, p=weights)

    # Set the weight of the option that was picked to half of the lowest weighted item
    weights[index] = min(weights) / 2

    # Double the weight of any option that wasn't picked
    weights *= 2

    return index, weights


def dissonant_counterpoint(items, skip=0):
    counter = Counter()

    len_items = len(items)
    indexes = np.arange(len_items)
    weights = np.ones(len_items)

    # Choose items until every item in `items` has been chosen at least `skip` times
    while not all([counter[i] >= skip for i in indexes]):
        index, weights = choose_and_reweight(indexes, weights)
        counter[index] += 1
        # print '\t'.join(['{}: {}'.format(i, counter[i]) for i in indexes])

    while True:
        index, weights = choose_and_reweight(indexes, weights)
        yield items[index]


# TODO: Use the function above to create the pattern of dissonant counterpoint
#       both in the absolute pitch classes used but also in the intervals
#       between consecutive pitches
