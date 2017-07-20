#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Test montecarlo simulation module

:author: Ben Johnston
:license: 3-Clause BSD

"""

# Imports
from unittest import TestCase, skip
from unittest.mock import patch
import statsbox.simulation.montecarlo as\
    montecarlo
import numpy as np


class TestMonteCarlo(TestCase):
    """Unit tests for MonteCarlo Simulation"""

    @classmethod
    def setUp(cls):
        """Setup method for the class"""
        cls.means = [10, 12] #mean values for 2 categories
        cls.sigma = [1, 2] # standard deviations for 2 categories
        cls.numsamples = [20, 30] # Number of samples for each category
        cls.walks = 10 # Number of walks

    def test_interface(self):
        """Test MonteCarlo class initialisation"""

        mc_sim = montecarlo.MonteCarlo(
            self.means,
            self.sigma,
            self.numsamples,
            self.walks
        )

        self.assertTrue(self.means, mc_sim.means)
        self.assertTrue(self.sigma, mc_sim.std_devs)
        self.assertTrue(self.numsamples, mc_sim.samples)
        self.assertTrue(self.walks, mc_sim.walks)
        self.assertTrue(hasattr(mc_sim, 'gen_dists'))

    def test_invalid_init(self):
        """Test MonteCarlo class raises assertion"""

        self.means += [10]

        with self.assertRaises(ValueError):
            montecarlo.MonteCarlo(
                self.means,
                self.sigma,
                self.numsamples,
                self.walks
            )

    def test_generate_distribution(self):
        """Test Montecarlo simulations generates the correct amount of data"""
        mc_sim = montecarlo.MonteCarlo(
            self.means,
            self.sigma,
            self.numsamples,
            self.walks
        )

        for sample, dist in enumerate(mc_sim.gen_dists()):
            dist_a, dist_b = dist

            with self.subTest(sample=sample):
                self.assertEqual(len(dist_a), self.numsamples[0])
                self.assertEqual(len(dist_b), self.numsamples[1])

        self.assertEqual(sample, self.walks - 1) # pylint: disable=W0631

    @patch('statsbox.simulation.montecarlo.MonteCarlo.gen_dists',
        side_effect=[[[
            np.array([12, 11, 10, 9, 8]),
            np.array([6, 5, 4, 3, 2]),
            np.array([6, 5, 4, 3, 2]),
        ]]])
    def test_apply_test_criteria(self, gen_dists_mock):
        """Test Montecarlo sim applying the test criteria - with false results"""

        # The actual mean, stdev etc variables do not matter because of the
        # patch
        mc_sim = montecarlo.MonteCarlo(
            self.means,
            self.sigma,
            self.numsamples,
            self.walks
        )

        # Apply to all test criteria
        results = mc_sim.apply_criteria(control_idx=2, criteria=4)

        false_neg, false_pos = results

        # Test the false negative results
        false_neg, minimum, mean, median, maximum, std_dev = false_neg

        self.assertEqual(minimum, 2)
        self.assertEqual(maximum, 3)
        self.assertEqual(mean, 2.5)
        self.assertEqual(median, 2.5)
        self.assertEqual(std_dev, 0.5)

        # Test the false positive results
        false_pos, minimum, mean, median, maximum, std_dev = false_pos
        self.assertEqual(minimum, 5)
        self.assertEqual(maximum, 6)
        self.assertEqual(mean, 5.5)
        self.assertEqual(median, 5.5)
        self.assertEqual(std_dev, 0.5)

    @patch('statsbox.simulation.montecarlo.MonteCarlo.gen_dists',
        side_effect=[[[
            np.array([12, 11, 10, 9, 8]),
            np.array([10, 9, 8, 7, 6]),
            np.array([5, 4, 3, 2, 1]),
        ]]])
    def test_apply_test_criteria_pass(self, gen_dists_mock):
        """Test Montecarlo sim applying the test criteria - with no false results"""

        # The actual mean, stdev etc variables do not matter because of the
        # patch
        mc_sim = montecarlo.MonteCarlo(
            self.means,
            self.sigma,
            self.numsamples,
            self.walks
        )

        # Apply to all test criteria
        results = mc_sim.apply_criteria(control_idx=2, criteria=5)

        false_neg, false_pos = results

        # Test the false negative results
        false_neg, minimum, mean, median, maximum, std_dev = false_neg

        self.assertTrue(np.isnan(minimum))
        self.assertTrue(np.isnan(maximum))
        self.assertTrue(np.isnan(mean))
        self.assertTrue(np.isnan(median))
        self.assertTrue(np.isnan(std_dev))

        # Test the false positive results
        false_pos, minimum, mean, median, maximum, std_dev = false_pos

        self.assertTrue(np.isnan(minimum))
        self.assertTrue(np.isnan(maximum))
        self.assertTrue(np.isnan(mean))
        self.assertTrue(np.isnan(median))
        self.assertTrue(np.isnan(std_dev))
