# pylint: disable=invalid-name
import unittest

import numpy as np
from mixtera.core.algo.aioli.aioli import AioliDynamicMixing


class TestAioliDynamicMixing(unittest.TestCase):
    def setUp(self):
        # Initialize the AdoDynamicMixing instance with default parameters
        self.aioli = AioliDynamicMixing()
        self.aioli.initial_mixture = np.array([0.4, 0.6])

        # Initialize counts and losses
        self.aioli.domain_count = len(self.aioli.initial_mixture)
        self.aioli.counts = np.zeros(self.aioli.domain_count, dtype=np.int64)
        self.aioli.losses = np.zeros(self.aioli.domain_count, dtype=np.float32)

    def test_init(self):
        dynamic_mixing = AioliDynamicMixing(
            eta=0.1,
            lp_rounds=1,
            lp_steps=10,
            update_steps=1000,
            aioli_normalize_A=False,
            aioli_diagonal=False,
            one_hot_factor=0.9,
            prior_steps=10,
        )
        self.assertEqual(dynamic_mixing.eta, 0.1)
        self.assertEqual(dynamic_mixing.lp_rounds, 1)
        self.assertEqual(dynamic_mixing.lp_steps, 10)
        self.assertEqual(dynamic_mixing.update_steps, 1000)
        self.assertEqual(dynamic_mixing.aioli_normalize_A, False)
        self.assertEqual(dynamic_mixing.one_hot_factor, 0.9)
        self.assertEqual(dynamic_mixing.aioli_diagonal, False)
