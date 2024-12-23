import unittest

import numpy as np
from mixtera.core.algo.dynamic_mixing.dynamic_mixing import DynamicMixingAlgorithm


class TestDynamicMixingAlgorithm(unittest.TestCase):
    class MinimalDynamicMixingAlgorithm(DynamicMixingAlgorithm):
        def calc_mixture(self, updated_at_client: bool) -> np.ndarray | None:
            return None

    def setUp(self):
        self.algorithm = self.MinimalDynamicMixingAlgorithm()

    def test_initial_state(self):
        self.assertEqual(self.algorithm.losses.size, 0)
        self.assertEqual(self.algorithm.counts.size, 0)
        self.assertTrue(np.allclose(self.algorithm.losses, np.array([], dtype=np.float32)))
        self.assertTrue(np.allclose(self.algorithm.counts, np.array([], dtype=np.int64)))
        self.assertTrue(np.allclose(self.algorithm.initial_mixture, np.array([-1], dtype=np.float32)))

    def test_process_losses(self):
        """Test process_losses method with initial data."""
        losses = np.array([0.1, 0.2, 0.3])
        counts = np.array([10, 20, 30])
        return_value = self.algorithm.process_losses(losses, counts, 0)
        # Check that the internal state was updated correctly
        self.assertTrue(np.allclose(self.algorithm.losses, losses))
        self.assertTrue(np.allclose(self.algorithm.counts, counts))
        # Since calc_mixture returns None, process_losses should return None
        self.assertIsNone(return_value)

    def test_process_losses_accumulates(self):
        """Test that process_losses accumulates losses and counts over multiple calls."""
        # First call
        losses1 = np.array([0.1, 0.2])
        counts1 = np.array([10, 20])
        self.algorithm.process_losses(losses1, counts1, 0)
        # Second call with same length arrays
        losses2 = np.array([0.3, 0.4])
        counts2 = np.array([30, 40])
        self.algorithm.process_losses(losses2, counts2, 0)
        # Expected accumulated losses and counts
        expected_losses = losses1 + losses2  # [0.4, 0.6]
        expected_counts = counts1 + counts2  # [40, 60]
        self.assertTrue(np.allclose(self.algorithm.losses, expected_losses))
        self.assertTrue(np.allclose(self.algorithm.counts, expected_counts))

    def test_process_losses_expands_arrays(self):
        """Test that process_losses expands internal arrays when new domains are added."""
        # First call with 2 domains
        losses1 = np.array([0.1, 0.2])
        counts1 = np.array([10, 20])
        self.algorithm.process_losses(losses1, counts1, 0)
        # Second call with 3 domains
        losses2 = np.array([0.3, 0.4, 0.5])
        counts2 = np.array([30, 40, 50])
        self.algorithm.process_losses(losses2, counts2, 0)
        # Expected losses and counts after expansion
        expected_losses = np.array([0.1 + 0.3, 0.2 + 0.4, 0 + 0.5])  # [0.4, 0.6, 0.5]
        expected_counts = np.array([10 + 30, 20 + 40, 0 + 50])  # [40, 60, 50]
        self.assertTrue(np.allclose(self.algorithm.losses, expected_losses))
        self.assertTrue(np.allclose(self.algorithm.counts, expected_counts))

    def test_reset_state(self):
        # Provide some data
        losses = np.array([0.1, 0.2, 0.3])
        counts = np.array([10, 20, 30])
        self.algorithm.process_losses(losses, counts, 0)
        # Now reset state
        self.algorithm.reset_state()
        # State should be reset
        self.assertEqual(self.algorithm.losses.size, 0)
        self.assertEqual(self.algorithm.counts.size, 0)
        self.assertTrue(np.allclose(self.algorithm.losses, np.array([], dtype=self.algorithm.losses.dtype)))
        self.assertTrue(np.allclose(self.algorithm.counts, np.array([], dtype=self.algorithm.counts.dtype)))

    def test_update_state_with_increasing_domain_sizes(self):
        """Test _update_state method with increasing domain sizes."""
        # First, internal arrays are empty
        self.assertEqual(len(self.algorithm.losses), 0)
        # Incoming data with 2 domains
        losses1 = np.array([0.1, 0.2])
        counts1 = np.array([1, 2])
        self.algorithm._update_state(losses1, counts1)
        self.assertTrue(np.allclose(self.algorithm.losses, losses1))
        self.assertTrue(np.allclose(self.algorithm.counts, counts1))
        # Incoming data with 3 domains
        losses2 = np.array([0.3, 0.4, 0.5])
        counts2 = np.array([3, 4, 5])
        self.algorithm._update_state(losses2, counts2)
        expected_losses = np.array([0.1 + 0.3, 0.2 + 0.4, 0.5])
        expected_counts = np.array([1 + 3, 2 + 4, 5])
        self.assertTrue(np.allclose(self.algorithm.losses, expected_losses))
        self.assertTrue(np.allclose(self.algorithm.counts, expected_counts))

    def test_update_state_with_decreasing_domain_sizes(self):
        # First, provide data with 3 domains
        losses1 = np.array([0.1, 0.2, 0.3])
        counts1 = np.array([1, 2, 3])
        self.algorithm._update_state(losses1, counts1)
        # Now provide data with 2 domains
        losses2 = np.array([0.4, 0.5])
        counts2 = np.array([4, 5])
        self.algorithm._update_state(losses2, counts2)
        expected_losses = np.array([0.1 + 0.4, 0.2 + 0.5, 0.3])
        expected_counts = np.array([1 + 4, 2 + 5, 3])
        self.assertTrue(np.allclose(self.algorithm.losses, expected_losses))
        self.assertTrue(np.allclose(self.algorithm.counts, expected_counts))
