# pylint: disable=invalid-name
import unittest

import numpy as np
from mixtera.core.algo.ado.ado import AdoDynamicMixing


class TestAdoDynamicMixing(unittest.TestCase):
    def setUp(self):
        # Initialize the AdoDynamicMixing instance with default parameters
        self.ado = AdoDynamicMixing()
        self.ado.initial_mixture = np.array([0.4, 0.6])
        # Initialize counts and losses
        num_domains = len(self.ado.initial_mixture)
        self.ado.counts = np.zeros(num_domains, dtype=np.int64)
        self.ado.losses = np.zeros(num_domains, dtype=np.float32)
        self.ado.mu_k = self.ado.initial_mixture.copy()

    def test_init(self):
        dynamic_mixing = AdoDynamicMixing(
            variant="vanilla",
            gamma1=0.1,
            gamma2=0.9,
            s=0.5,
            delta_min=0.01,
            scaling_law_update_interval=1000,
            subsampling_interval=10,
            ignore_initial_steps=500,
        )
        self.assertEqual(dynamic_mixing.variant, "vanilla")
        self.assertEqual(dynamic_mixing.gamma1, 0.1)
        self.assertEqual(dynamic_mixing.gamma2, 0.9)
        self.assertEqual(dynamic_mixing.s, 0.5)
        self.assertEqual(dynamic_mixing.delta_min, 0.01)
        self.assertEqual(dynamic_mixing.scaling_law_update_interval, 1000)
        self.assertEqual(dynamic_mixing.ignore_initial_steps, 500)
        self.assertIsNone(dynamic_mixing.mu_k)
        self.assertIsNone(dynamic_mixing.h_t)
        self.assertIsNone(dynamic_mixing.pi_t)
        self.assertIsNone(dynamic_mixing.pi_t_minus_1)
        self.assertIsNone(dynamic_mixing.pi_bar_t_minus_1)
        self.assertEqual(dynamic_mixing.total_steps, 0)

    def test_calc_mixture_warmup(self):
        # Test calc_mixture method during warm-up period
        self.ado.total_steps = self.ado.ignore_initial_steps - 1
        mixture = self.ado.calc_mixture(updated_at_client=False)
        # Should return None during warm-up
        self.assertIsNone(mixture)

    def test_calc_mixture_after_warmup(self):
        # Test calc_mixture method after warm-up
        self.ado.total_steps = self.ado.ignore_initial_steps + self.ado.scaling_law_update_interval + 1
        self.ado.counts = np.array([100, 200])
        self.ado.losses = np.array([1.0, 0.5])
        # Mock per_step_counts and per_step_losses
        self.ado.per_step_counts = [np.array([10, 20]) for _ in range(1000)]
        self.ado.per_step_losses = [np.array([1.0, 0.5]) for _ in range(1000)]
        # Mock scaling_law_params
        self.ado.scaling_law_params = np.array([[np.log(1.0), np.log(0.1), 0.5], [np.log(0.5), np.log(0.05), 0.4]])
        # Initialize h_t
        self.ado.h_t = np.array([0.5, 0.5])
        self.ado.pi_t = np.array([0.4, 0.6])
        # Calculate mixture
        mixture = self.ado.calc_mixture(updated_at_client=True)
        # Check that mixture is not None
        self.assertIsNotNone(mixture)
        # Check that mixture sums to 1
        self.assertAlmostEqual(np.sum(mixture), 1.0)
        # Check the length of the mixture
        self.assertEqual(len(mixture), 2)

    def test_update_h_t(self):
        # Test update_h_t method
        self.ado.h_t = np.array([0.3, 0.7])
        self.ado.pi_t_minus_1 = np.array([0.4, 0.6])
        self.ado.gamma1 = 0.1
        self.ado.update_h_t()
        expected_h_t = 0.1 * np.array([0.4, 0.6]) + 0.9 * np.array([0.3, 0.7])
        np.testing.assert_array_almost_equal(self.ado.h_t, expected_h_t)

    def test_update_h_t_adjusted_v2(self):
        # Test update_h_t with variant 'adjusted_v2' and elapsed_steps
        self.ado.variant = "adjusted_v2"
        self.ado.h_t = np.array([0.3, 0.7])
        self.ado.pi_t_minus_1 = np.array([0.4, 0.6])
        self.ado.gamma1 = 0.1
        self.ado.total_steps = 105
        self.ado.last_update_step = 100
        elapsed_steps = self.ado.total_steps - self.ado.last_update_step

        prev_h_t = self.ado.h_t.copy()

        self.ado.update_h_t(elapsed_steps=elapsed_steps)
        adjusted_gamma1 = 1 - (1 - self.ado.gamma1) ** elapsed_steps
        expected_h_t = adjusted_gamma1 * self.ado.pi_t_minus_1 + (1 - adjusted_gamma1) * prev_h_t

        np.testing.assert_array_almost_equal(self.ado.h_t, expected_h_t)

    def test_fit_scaling_laws(self):
        # Test fit_scaling_laws method
        # Mock per_step_counts and per_step_losses with synthetic data
        steps = 1000
        counts = np.cumsum(np.ones((steps, 2)), axis=0)
        beta_k = np.array([1.0, 0.5])
        epsilon_k = np.array([0.1, 0.05])
        alpha_k = np.array([0.5, 0.4])
        losses = epsilon_k + beta_k / counts**alpha_k
        self.ado.per_step_counts = counts.tolist()
        self.ado.per_step_losses = losses.tolist()
        # Call fit_scaling_laws
        self.ado.fit_scaling_laws()
        # Check that scaling_law_params have been set
        self.assertIsNotNone(self.ado.scaling_law_params)
        self.assertEqual(self.ado.scaling_law_params.shape, (2, 3))

    def test_scaling_law_loss(self):
        # Test scaling_law_loss static method
        counts_k = np.array([1, 2, 3, 4, 5])
        losses_k = 0.1 + 1.0 / counts_k**0.5
        params = (np.log(1.0), np.log(0.1), 0.5)
        loss = AdoDynamicMixing.scaling_law_loss(params, counts_k, losses_k)
        # Check that the loss is a finite positive number
        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(loss, 0)

    def test_compute_loss_derivative(self):
        # Test compute_loss_derivative method
        self.ado.scaling_law_params = np.array([[np.log(1.0), np.log(0.1), 0.5], [np.log(0.5), np.log(0.05), 0.4]])
        self.ado.counts = np.array([10, 20])
        dL_dn = self.ado.compute_loss_derivative()
        # Check that the derivative array has the correct length
        self.assertEqual(len(dL_dn), 2)
        # Check that the derivatives are finite and positive
        self.assertTrue(np.all(np.isfinite(dL_dn)))
        self.assertTrue(np.all(dL_dn >= 0))

    def test_compute_rho_t(self):
        # Test compute_rho_t method
        self.ado.mu_k = np.array([0.4, 0.6])
        self.ado.h_t = np.array([0.5, 0.5])
        self.ado.s = 0.5
        dL_dn = np.array([0.1, 0.2])
        self.ado.compute_rho_t(dL_dn)
        expected_lambda_k_t = self.ado.h_t**self.ado.s
        rho_num = self.ado.mu_k * expected_lambda_k_t * dL_dn
        rho_den = np.sum(rho_num)
        expected_rho_t = rho_num / rho_den if rho_den > 0 else self.ado.mu_k / len(self.ado.counts)
        np.testing.assert_array_almost_equal(self.ado.rho_t, expected_rho_t)

    def test_update_pi_t(self):
        # Test update_pi_t method
        self.ado.gamma2 = 0.9
        self.ado.rho_t = np.array([0.6, 0.4])
        self.ado.pi_bar_t_minus_1 = np.array([0.5, 0.5])
        self.ado.pi_t = np.array([0.4, 0.6])
        assert self.ado.pi_t is not None
        self.ado.update_pi_t()
        expected_pi_t = self.ado.gamma2 * self.ado.rho_t + (1 - self.ado.gamma2) * self.ado.pi_bar_t_minus_1
        expected_pi_t /= np.sum(expected_pi_t)
        np.testing.assert_array_almost_equal(self.ado.pi_t, expected_pi_t)

    def test_enforce_min_probability(self):
        # Test enforce_min_probability method
        self.ado.pi_t = np.array([0.05, 0.95])
        self.ado.counts = np.array([10, 0])
        self.ado.delta_min = 0.1
        self.ado.mu_k = np.array([0.4, 0.6])
        self.ado.enforce_min_probability()
        # The first domain has counts > 0 and should have pi_t >= delta_min with some slack due to normlaization
        self.assertGreaterEqual(self.ado.pi_t[0], self.ado.delta_min - 0.01)
        # Pi_t should sum to 1
        self.assertAlmostEqual(np.sum(self.ado.pi_t), 1.0)

    def test_update_pi_bar_t(self):
        # Test update_pi_bar_t method
        self.ado.total_steps = 10
        self.ado.pi_bar_t_minus_1 = np.array([0.5, 0.5])
        self.ado.rho_t = np.array([0.6, 0.4])
        self.ado.pi_t = np.array([0.6, 0.4])
        prev_pi_bar_t_minus_1 = self.ado.pi_bar_t_minus_1.copy()

        self.ado.update_pi_bar_t()
        weight = 1.0 / (self.ado.total_steps + 1.0)
        expected_pi_bar_t = weight * self.ado.pi_t + (1 - weight) * prev_pi_bar_t_minus_1
        expected_pi_bar_t /= np.sum(expected_pi_bar_t)
        np.testing.assert_array_almost_equal(self.ado.pi_bar_t_minus_1, expected_pi_bar_t)

    def test__update_state(self):
        # Test _update_state method
        # Initialize counts and losses
        self.ado.counts = np.array([100, 200])
        self.ado.losses = np.array([10.0, 5.0])
        # Existing per_step_counts and per_step_losses
        self.ado.per_step_counts = [np.array([10, 20])]
        self.ado.per_step_losses = [np.array([1.0, 0.5])]
        # New losses and counts with additional domain
        new_losses = np.array([12.0, 6.0, 3.0])
        new_counts = np.array([120, 240, 360])
        # Call _update_state
        self.ado._update_state(new_losses, new_counts)
        # Check that counts and losses have been expanded
        self.assertEqual(len(self.ado.counts), 3)
        self.assertEqual(len(self.ado.losses), 3)
        # Check that per_step_counts and per_step_losses have correct shapes
        self.assertEqual(len(self.ado.per_step_counts[-1]), 3)
        self.assertEqual(len(self.ado.per_step_losses[-1]), 3)
        # Check that per_step_losses are average losses per sample
        expected_per_step_losses = new_losses / new_counts
        np.testing.assert_array_almost_equal(self.ado.per_step_losses[-1], expected_per_step_losses)

    def test_update_method(self):
        """
        Test the update process, ensuring that losses and counts are kept track correctly.
        """
        # Initial counts and losses
        initial_counts = np.array([100, 200])
        initial_losses = np.array([10.0, 5.0])
        self.ado.counts = initial_counts.copy()
        self.ado.losses = initial_losses.copy()

        # Existing per_step_counts and per_step_losses
        self.ado.per_step_counts = [np.array([50, 100])]
        self.ado.per_step_losses = [np.array([0.1, 0.05])]  # average losses per sample

        # New losses and counts
        new_counts = np.array([50, 100])
        new_losses = np.array([5.0, 2.5])
        mixture_id = 0  # Assuming initial mixture_id

        # Expected counts and losses after update
        expected_counts = initial_counts + new_counts
        expected_losses = initial_losses + new_losses

        # Expected per-step counts and losses
        expected_per_step_counts = self.ado.per_step_counts + [new_counts]
        expected_per_step_losses = self.ado.per_step_losses + [new_losses / new_counts]

        # Process losses
        new_mixture = self.ado.process_losses(new_losses, new_counts, mixture_id)

        # Check that counts and losses have been updated correctly
        np.testing.assert_array_equal(self.ado.counts, expected_counts)
        np.testing.assert_array_equal(self.ado.losses, expected_losses)

        # Since calc_mixture returns None by default, check that new_mixture is None for now
        self.assertIsNone(new_mixture)

        # Check that per_step_counts and per_step_losses have been appended correctly
        self.assertEqual(len(self.ado.per_step_counts), 2)
        self.assertEqual(len(self.ado.per_step_losses), 2)
        np.testing.assert_array_equal(self.ado.per_step_counts[-1], new_counts)
        np.testing.assert_array_almost_equal(self.ado.per_step_losses[-1], new_losses / new_counts)

        # Additionally, we can verify that the accumulated per-step counts and losses are as expected
        for idx, (counts, losses) in enumerate(zip(expected_per_step_counts, expected_per_step_losses)):
            np.testing.assert_array_equal(self.ado.per_step_counts[idx], counts)
            np.testing.assert_array_almost_equal(self.ado.per_step_losses[idx], losses)

        # Verify that total_steps has been incremented
        self.assertEqual(self.ado.total_steps, 1)
