"""See https://math.dartmouth.edu/~ahb/papers/dc.pdf"""

from collections import Counter

import numpy as np


def weighted_choice(indexes, weights):
    # Make weights sum to 1.0
    weights /= weights.sum()

    # Weighted choice
    index = np.random.choice(indexes, p=weights)

    return index, weights


def reweight(index, weights, multiplier=2):
    multiplier = float(multiplier)

    # Double the weight of all the options
    weights *= multiplier

    # Set the weight of the option that was picked to half of the lowest weighted item
    weights[index] = weights.min() / multiplier

    return weights


def dissonant_counterpoint(items, skip=0, multiplier=2):
    counter = Counter()

    len_items = len(items)
    indexes = np.arange(len_items)
    weights = np.ones(len_items)

    # Choose items until every item in `items` has been chosen at least `skip` times
    while not all([counter[i] >= skip for i in indexes]):
        index, weights = weighted_choice(indexes, weights)
        weights = reweight(index, weights, multiplier=multiplier)

        counter[index] += 1
        # print '\t'.join(['{}: {}'.format(i, counter[i]) for i in indexes])

    while True:
        index, weights = weighted_choice(indexes, weights)
        weights = reweight(index, weights, multiplier=multiplier)

        yield items[index]
