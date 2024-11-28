

import numpy as np
from mixtera.core.algo.dynamic_mixing.dynamic_mixing import DynamicMixingAlgorithm


class SimpleAveragingAlgorithm(DynamicMixingAlgorithm):
    """
    Simple averaging algorithm where the mixture coefficients are higher if the average loss is higher.

    This algorithm computes the mixture coefficients proportional to the average loss per domain,
    considering only domains with observed counts.
    """

    def calc_mixture(self) -> np.ndarray | None:
        """
        Computes the updated mixture coefficients based on the accumulated losses and counts.

        Returns:
            A numpy array representing the new mixture coefficients, or None if no update is available.
        """
        # Only consider domains with counts > 0
        mask = self.counts > 0
        if not np.any(mask):
            return None

        # Compute average loss per domain
        avg_losses = np.zeros_like(self.losses, dtype=np.float32)
        avg_losses[mask] = self.losses[mask] / self.counts[mask]

        # Compute mixture coefficients proportional to average losses
        total_avg_loss = np.sum(avg_losses[mask])
        if total_avg_loss == 0:
            # If total average loss is zero, assign equal mixture weights to domains with counts > 0
            mixture = np.zeros_like(self.losses, dtype=np.float32)
            mixture[mask] = 1.0 / np.sum(mask)
        else:
            # Normalize average losses to sum to 1
            mixture = np.zeros_like(self.losses, dtype=np.float32)
            mixture[mask] = avg_losses[mask] / total_avg_loss

        # Ensure mixture coefficients sum to 1
        mixture_sum = np.sum(mixture)
        if mixture_sum != 0:
            mixture /= mixture_sum

        return mixture
