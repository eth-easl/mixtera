# pylint: disable=invalid-name
import unittest

import numpy as np
from mixtera.core.algo.aioli.aioli import AioliDynamicMixing


class TestAioliDynamicMixing(unittest.TestCase):
    def setUp(self):
        # Initialize the AioliDynamicMixing instance with default parameters
        self.aioli = AioliDynamicMixing()
        self.aioli.initial_mixture = np.array([0.4, 0.6])
        self.aioli.weights = self.aioli.initial_mixture
        self.aioli.last_training_steps = 0
        self.aioli.prior_steps = 50
        self.aioli.lp_rounds = 1
        self.aioli.lp_steps = 5
        self.aioli.update_steps = 20
        self.aioli.eta = 0.1

        # Initialize counts and losses
        self.aioli.domain_count = len(self.aioli.initial_mixture)
        self.aioli.graph = np.zeros((self.aioli.domain_count, self.aioli.domain_count))
        self.aioli.counts = np.zeros(self.aioli.domain_count, dtype=np.int64)
        self.aioli.losses = np.zeros(self.aioli.domain_count, dtype=np.float32)

    def test_init(self):
        dynamic_mixing = AioliDynamicMixing(
            eta=0.1,
            lp_rounds=1,
            lp_steps=10,
            update_steps=1000,
            aioli_normalize_a=False,
            aioli_diagonal=False,
            one_hot_factor=0.9,
            prior_steps=10,
        )
        self.assertEqual(dynamic_mixing.eta, 0.1)
        self.assertEqual(dynamic_mixing.lp_rounds, 1)
        self.assertEqual(dynamic_mixing.lp_steps, 10)
        self.assertEqual(dynamic_mixing.update_steps, 1000)
        self.assertEqual(dynamic_mixing.aioli_normalize_a, False)
        self.assertEqual(dynamic_mixing.one_hot_factor, 0.9)
        self.assertEqual(dynamic_mixing.aioli_diagonal, False)

    def test_learn_params_subroutine(self):
        self.aioli.graph = np.array([[-0.5, 0.1], [0.0, -0.2]])
        self.aioli.one_hot_factor = 0.9
        self.aioli.aioli_diagonal = False

        weight_matrix = np.zeros((self.aioli.domain_count, self.aioli.domain_count))
        for i in range(self.aioli.domain_count):
            weight_row = (
                np.ones(self.aioli.domain_count) * (1 - self.aioli.one_hot_factor) / (self.aioli.domain_count - 1)
            )
            weight_row[i] = self.aioli.one_hot_factor
            weight_matrix[i] = weight_row

        expected_graph = np.zeros((self.aioli.domain_count, self.aioli.domain_count))
        for i, row in enumerate(self.aioli.graph):
            expected_graph[i] = np.linalg.solve(weight_matrix, row)

        self.aioli.learn_params_subroutine()
        np.testing.assert_array_almost_equal(self.aioli.graph, expected_graph)

    def test_calc_mixture_no_update(self):
        mixture = self.aioli.calc_mixture(updated_at_client=False)
        np.testing.assert_almost_equal(self.aioli.initial_mixture, mixture)

    def test_calc_mixture_perturbed_domains(self):
        self.aioli.one_hot_factor = 0.9
        self.aioli.last_received_mixture = 1
        mixture = self.aioli.calc_mixture(updated_at_client=True)
        expected_mixture = np.array([0.1, 0.9])

        np.testing.assert_array_almost_equal(expected_mixture, mixture)

    def test_calc_mixture_new_weights(self):
        self.aioli.graph = np.array([[-0.5, 0.1], [0.0, -0.2]])
        initial_mixture = np.array([0.4, 0.6])

        self.aioli.weights = initial_mixture
        self.aioli.one_hot_factor = 0.9
        self.aioli.last_received_mixture = -1
        self.aioli.aioli_diagonal = False

        counts = np.array([100, 200])
        losses = np.array([10.0, 5.0])

        mixture = self.aioli.process_losses(losses, counts, 0)
        expected_mixture = np.array([0.9, 0.1])
        np.testing.assert_array_almost_equal(expected_mixture, mixture)

        losses = np.array([9.0, 6.0])
        mixture = self.aioli.process_losses(losses, counts, 1)
        expected_mixture = np.array([0.1, 0.9])
        np.testing.assert_array_almost_equal(expected_mixture, mixture)

        losses = np.array([8.0, 7.0])
        mixture = self.aioli.process_losses(losses, counts, 2)
        expected_mixture = np.multiply(self.aioli.weights, np.exp(self.aioli.eta * self.aioli.graph.sum(axis=0)))
        expected_mixture = expected_mixture / sum(expected_mixture)
        np.testing.assert_array_almost_equal(expected_mixture, mixture)
