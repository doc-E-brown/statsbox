#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Class for completing montecarlo simeanlations



:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
from time import time
import numpy as np


class MonteCarlo(object):
    """Class for completing montecarlo simeanlation

    samplesotes
    ----------

    * Currently assumes each category to be simeanlated
    from a normal (Gaussian) distribution

    """

    def __init__(self, means=[], std_dev=[], samples=[], walks=0):
        """Initialse the class object with simeanlation parameters

        Parameters
        ----------

        means: iterable
            Mean values of the simeanlation categories

        std_dev: iterable
            Standard deviation values of the simeanlation categories

        samples: iterable
            samplesumber of random values to generate for each category

        walks: int
            samplesumber of times simeanlation is repeated

        Returns
        ----------
        None

        """

        if ((len(means) != len(std_dev)) or
                (len(means) != len(samples))):
            raise ValueError(
                "number of categories for means, std_dev and samples are unequal")

        self.means = means
        self.std_devs = std_dev # pylint: disable=C0103
        self.samples = samples
        self.walks = walks
        self.categories = len(means)
        # Seed random number generator
        self._rng = np.random.RandomState(int(time())) # pylint: disable=E1101

    def gen_dists(self):
        """Generate distibutions for all categories over number of walks"""

        # Iterate through specified number of walks
        for _ in range(self.walks):

            # Iterate through mean and standard deviations
            dists = []
            for mean, std_dev, samples in zip(self.means, self.std_devs, self.samples):
                dists.append(self._rng.normal(mean, std_dev, samples)) # pylint: disable=E1101

            yield dists

    def apply_criteria(self, control_idx=-1, criteria = 0, test_cri='__lt__'):
        """Apply criteria to compare category to the control

        For a random distribution as determined by the parameters of the montecarlo
        simulation, calculate distributions of false negative and false positive samples.    

        .. math::
            p(FalseNeg) = \frac{\sum_{i=0}^{N}(X_i < c | i \notin control)}{N}
            p(FalsePos) = \frac{\sum_{i=0}^{N}(X_i > c | i \in control)}{N}

        Parameters
        ----------

        control_idx: integer
            Index identifying the control category in the mean and standard deviation lists
            (default: -1)

        criteria: integer or float 
            The criteria to test the separation between the experimental and control groups
            (default: 0)

        test: string
            The test to be used to apply to the test criteria and determine if a sample is
            considered a PASS or FAIL. If test='__lt__' then a sample is considered to be a
            PASS if it is less than the criteria.
            (default: __lt__)

        Returns
        ----------

        A dictionary summarising the data produced for the False Negative (FALSE_NEG) and False Positive
        (FALSE_POS) scenarios with the applied test criteria and control category.

        For each key in the dictionary a list of the following values is produced: 

        minimum: float
            The minimum value of FALSE_NEG or FALSE_POS distributions 

        mean: float
            The mean value of FALSE_NEG or FALSE_POS distributions 

        median: float
            The median value of FALSE_NEG or FALSE_POS distributions 

        maximum: float
            The maximum value of FALSE_NEG or FALSE_POS distributions 

        std: float
            The standard deviation value of FALSE_NEG or FALSE_POS distributions 

        {
            'FALSE_NEG': [0, 1, 2, 3, 0.1],
            'FALSE_POS': [4, 5, 6, 7, 0.2],
        }

        """

        # There are only two groups, PASS or FAIL
        FAIL = 0
        PASS = 1
        all_dists = [np.array([])] * 2 

        # Compute random samples
        for dists in self.gen_dists():
            for idx, dist in enumerate(dists):
                all_dists[(idx == control_idx) + FAIL] =\
                    np.concatenate((all_dists[(idx == control_idx) + FAIL],
                    dist))

        # Compute false positive and negative calculations
        false_neg_idx = np.where(all_dists[FAIL] < criteria)[0]
        false_pos_idx = np.where(all_dists[PASS] > criteria)[0]

        # Extract false negatives
        if false_neg_idx.size != 0:
            false_neg = np.sort(all_dists[FAIL][false_neg_idx])
            false_neg = (
                false_neg,
                np.min(false_neg),
                np.mean(false_neg),
                np.median(false_neg),
                np.max(false_neg),
                np.std(false_neg),
            )

        else:
            # There are no false negatives so no distribution
            false_neg = np.array([])
            false_neg = (
                false_neg,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan
            )

        # Extract false positives 
        if false_pos_idx.size != 0:
            false_pos = np.sort(all_dists[PASS][false_pos_idx])
            false_pos = (
                false_pos,
                np.min(false_pos),
                np.mean(false_pos),
                np.median(false_pos),
                np.max(false_pos),
                np.std(false_pos),
            )

        else:
            false_pos = np.array([])
            false_pos = (
                false_pos,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan
            )
        return (false_neg, false_pos)
