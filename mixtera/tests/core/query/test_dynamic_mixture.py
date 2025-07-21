import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from mixtera.core.algo.dynamic_mixing.dynamic_mixing import DynamicMixingAlgorithm
from mixtera.core.query.mixture.dynamic_mixture import DynamicMixture
from mixtera.core.query.mixture.mixture import Mixture
from mixtera.core.query.mixture.mixture_key import MixtureKey


class TestDynamicMixture(unittest.TestCase):
    def setUp(self):
        self.chunk_size = 10
        # Mock initial_mixture
        self.initial_mixture = MagicMock(spec=Mixture)
        self.initial_mixture.chunk_size = self.chunk_size  # Set same chunk_size to avoid warnings
        # Mock mixing_alg
        self.mixing_alg = MagicMock(spec=DynamicMixingAlgorithm)
        # Create DynamicMixture instance
        self.dynamic_mixture = DynamicMixture(self.chunk_size, self.initial_mixture, self.mixing_alg)

    def test_init_adjusts_chunk_size(self):
        initial_mixture = MagicMock(spec=Mixture)
        initial_mixture.chunk_size = 5
        dynamic_mixture = DynamicMixture(self.chunk_size, initial_mixture, self.mixing_alg)
        self.assertEqual(initial_mixture.chunk_size, self.chunk_size)

        self.assertEqual(dynamic_mixture._current_mixture, initial_mixture)
        self.assertEqual(dynamic_mixture._mixing_alg, self.mixing_alg)
        self.assertFalse(dynamic_mixture._informed_alg)
        self.assertIsNone(dynamic_mixture._key_id_map)
        self.assertIsNone(dynamic_mixture._id_key_map)

    def test_init_no_adjustment_needed(self):
        self.assertEqual(self.dynamic_mixture._current_mixture, self.initial_mixture)
        self.assertEqual(self.dynamic_mixture._mixing_alg, self.mixing_alg)
        self.assertFalse(self.dynamic_mixture._informed_alg)
        self.assertIsNone(self.dynamic_mixture._key_id_map)
        self.assertIsNone(self.dynamic_mixture._id_key_map)

    def test_mixture_in_rows(self):
        # Mock the return value of initial_mixture.mixture_in_rows
        self.initial_mixture.mixture_in_rows.return_value = {"key1": 1, "key2": 2}
        result = self.dynamic_mixture.mixture_in_rows()
        self.initial_mixture.mixture_in_rows.assert_called_once()
        self.assertEqual(result, {"key1": 1, "key2": 2})

    def test_inform_about_id_map_first_time(self):
        key_id_map = {
            MixtureKey({"key1": ["val1"]}): 0,
            MixtureKey({"key2": ["val2"]}): 1,
            MixtureKey({"key3": ["val3"]}): 2,
        }
        mixture_in_rows = {
            MixtureKey({"key1": ["val1"]}): 0.2,
            MixtureKey({"key2": ["val2"]}): 0.5,
            MixtureKey({"key3": ["val3"]}): 0.3,
        }
        self.assertFalse(self.dynamic_mixture._informed_alg)
        self.initial_mixture.mixture_in_rows.return_value = mixture_in_rows
        self.mixing_alg.initial_mixture = np.array([], dtype=np.float32)
        self.dynamic_mixture.process_id_map(key_id_map)
        self.assertEqual(self.dynamic_mixture._key_id_map, key_id_map)
        expected_id_key_map = {
            0: MixtureKey({"key1": ["val1"]}),
            1: MixtureKey({"key2": ["val2"]}),
            2: MixtureKey({"key3": ["val3"]}),
        }
        self.assertEqual(self.dynamic_mixture._id_key_map, expected_id_key_map)

        expected_initial_mix = np.array([0.2, 0.5, 0.3], dtype=float)
        np.testing.assert_allclose(self.mixing_alg.initial_mixture, expected_initial_mix)
        self.assertTrue(self.dynamic_mixture._informed_alg)

    def test_process_losses_with_update(self):
        self.dynamic_mixture._key_id_map = {MixtureKey({"key1": ["val1"]}): 0, MixtureKey({"key2": ["val2"]}): 1}
        self.dynamic_mixture._id_key_map = {0: MixtureKey({"key1": ["val1"]}), 1: MixtureKey({"key2": ["val2"]})}
        self.dynamic_mixture._informed_alg = True

        losses = np.array([0.1, 0.2])
        counts = np.array([100, 200])

        new_mixture_np = np.array([0.4, 0.6])
        self.mixing_alg.process_losses.return_value = new_mixture_np
        with patch("mixtera.core.query.mixture.dynamic_mixture.StaticMixture") as mock_static_mixture:
            instance = mock_static_mixture.return_value
            self.dynamic_mixture._process_losses(losses, counts, 0)
            self.mixing_alg.process_losses.assert_called_once_with(losses, counts, 0)
            mock_static_mixture.assert_called_once_with(
                self.chunk_size, {MixtureKey({"key1": ["val1"]}): 0.4, MixtureKey({"key2": ["val2"]}): 0.6}, strict=True
            )
            self.assertEqual(self.dynamic_mixture._current_mixture, instance)

    def test_process_losses_no_update(self):
        self.dynamic_mixture._key_id_map = {MixtureKey({"key1": ["val1"]}): 0, MixtureKey({"key2": ["val2"]}): 1}
        self.dynamic_mixture._id_key_map = {0: MixtureKey({"key1": ["val1"]}), 1: MixtureKey({"key2": ["val2"]})}
        self.dynamic_mixture._informed_alg = True
        losses = np.array([0.1, 0.2])
        counts = np.array([100, 200])
        self.mixing_alg.process_losses.return_value = None
        with patch("mixtera.core.query.mixture.dynamic_mixture.StaticMixture") as mock_static_mixture:
            self.dynamic_mixture._process_losses(losses, counts, 0)
            self.mixing_alg.process_losses.assert_called_once_with(losses, counts, 0)
            # StaticMixture should not be called since there's no update
            mock_static_mixture.assert_not_called()
            # _current_mixture should remain unchanged
            self.assertEqual(self.dynamic_mixture._current_mixture, self.initial_mixture)

    def test_process_losses_mixture_sum_not_one(self):
        self.dynamic_mixture._key_id_map = {MixtureKey({"key1": ["val1"]}): 0}
        self.dynamic_mixture._id_key_map = {0: MixtureKey({"key1": ["val1"]})}
        self.dynamic_mixture._informed_alg = True
        losses = np.array([0.1])
        counts = np.array([100])
        # Return a mixture that does not sum to 1
        self.mixing_alg.process_losses.return_value = np.array([0.5])
        # Should raise an AssertionError since the mixture sum is not close to 1
        with self.assertRaises(AssertionError):
            self.dynamic_mixture._process_losses(losses, counts, 0)

    def test_end_to_end_dynamic_update(self):
        # Set up initial mixture
        key_id_map = {
            MixtureKey({"key1": ["val1"]}): 0,
            MixtureKey({"key2": ["val2"]}): 1,
            MixtureKey({"key3": ["val3"]}): 2,
        }
        mixture_in_rows = {
            MixtureKey({"key1": ["val1"]}): 0.2,
            MixtureKey({"key2": ["val2"]}): 0.5,
            MixtureKey({"key3": ["val3"]}): 0.3,
        }
        self.initial_mixture.mixture_in_rows.return_value = mixture_in_rows
        self.mixing_alg.initial_mixture = np.array([], dtype=np.float32)
        self.dynamic_mixture.process_id_map(key_id_map)

        self.assertEqual(self.dynamic_mixture.mixture_in_rows(), mixture_in_rows)

        # First process_losses call - mixing algorithm returns new mixture
        self.dynamic_mixture._informed_alg = True
        losses_step1 = np.array([0.1, 0.2, 0.3])
        counts_step1 = np.array([100, 200, 300])
        new_mixture_np_step1 = np.array([0.3, 0.4, 0.3])
        self.mixing_alg.process_losses.return_value = new_mixture_np_step1

        with patch("mixtera.core.query.mixture.dynamic_mixture.StaticMixture") as mock_static_mixture:
            instance_step1 = mock_static_mixture.return_value
            self.dynamic_mixture._process_losses(losses_step1, counts_step1, 0)
            # Ensure process_losses called with correct parameters
            self.mixing_alg.process_losses.assert_called_with(losses_step1, counts_step1, 0)
            # Ensure StaticMixture created with new mixture
            mock_static_mixture.assert_called_with(
                self.chunk_size,
                {
                    MixtureKey({"key1": ["val1"]}): 0.3,
                    MixtureKey({"key2": ["val2"]}): 0.4,
                    MixtureKey({"key3": ["val3"]}): 0.3,
                },
                strict=True,
            )
            # Update internal _current_mixture
            self.assertEqual(self.dynamic_mixture._current_mixture, instance_step1)

        # Second process_losses call - mixing algorithm returns None (no update)
        self.mixing_alg.process_losses.return_value = None
        losses_step2 = np.array([0.15, 0.25, 0.35])
        counts_step2 = np.array([150, 250, 350])

        with patch("mixtera.core.query.mixture.dynamic_mixture.StaticMixture") as mock_static_mixture:
            self.dynamic_mixture._process_losses(losses_step2, counts_step2, 0)
            # Ensure process_losses called with correct parameters
            self.mixing_alg.process_losses.assert_called_with(losses_step2, counts_step2, 0)
            # StaticMixture should not be called since there's no update
            mock_static_mixture.assert_not_called()
            # _current_mixture should remain as instance_step1
            self.assertEqual(self.dynamic_mixture._current_mixture, instance_step1)

        # Third process_losses call - mixing algorithm returns another new mixture
        new_mixture_np_step3 = np.array([0.2, 0.5, 0.3])
        self.mixing_alg.process_losses.return_value = new_mixture_np_step3
        losses_step3 = np.array([0.2, 0.3, 0.4])
        counts_step3 = np.array([200, 300, 400])

        with patch("mixtera.core.query.mixture.dynamic_mixture.StaticMixture") as mock_static_mixture:
            instance_step3 = mock_static_mixture.return_value
            self.dynamic_mixture._process_losses(losses_step3, counts_step3, 0)
            # Ensure process_losses called with correct parameters
            self.mixing_alg.process_losses.assert_called_with(losses_step3, counts_step3, 0)
            # Ensure StaticMixture created with new mixture
            mock_static_mixture.assert_called_with(
                self.chunk_size,
                {
                    MixtureKey({"key1": ["val1"]}): 0.2,
                    MixtureKey({"key2": ["val2"]}): 0.5,
                    MixtureKey({"key3": ["val3"]}): 0.3,
                },
                strict=True,
            )
            # Update internal _current_mixture
            self.assertEqual(self.dynamic_mixture._current_mixture, instance_step3)

        # Check that the mixing algorithm's process_losses was called the correct number of times
        self.assertEqual(self.mixing_alg.process_losses.call_count, 3)
