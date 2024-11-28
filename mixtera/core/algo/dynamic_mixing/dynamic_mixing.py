from abc import ABC, abstractmethod

import numpy as np


class DynamicMixingAlgorithm(ABC):
    """
    Abstract base class for dynamic mixing algorithms.

    This class defines the interface for dynamic mixing algorithms and provides support for accumulating
    losses and counts across training steps, even when the number of domains (or mixture components)
    changes over time.

    Subclasses should implement the `calc_mixture` method, which computes the updated mixture
    coefficients based on the accumulated losses and counts.
    """

    def __init__(self) -> None:
        """
        Initializes the dynamic mixing algorithm, setting up accumulators for losses and counts.
        """
        self.losses = np.array([], dtype=np.float32)
        self.counts = np.array([], dtype=np.int64)
        self.initial_mixture = np.array([-1], dtype=np.float32)

    def process_losses(self, losses: np.ndarray, counts: np.ndarray) -> np.ndarray | None:
        """
        Receives arrays of losses and counts, accumulates them, and returns the updated mixture if available.

        Args:
            losses: A numpy array of losses per domain. Each index corresponds to a specific domain.
            counts: A numpy array of counts (e.g., number of tokens) per domain.

        Returns:
            A numpy array representing the new mixture coefficients, or None if no update is available.
        """
        self._update_state(losses, counts)
        return self.calc_mixture()

    def _update_state(self, losses: np.ndarray, counts: np.ndarray) -> None:
        """
        Accumulates the losses and counts, adjusting internal arrays as needed to accommodate new domains.

        Args:
            losses: A numpy array of losses per domain.
            counts: A numpy array of counts per domain.
        """
        num_incoming_domains = len(losses)
        num_internal_domains = len(self.losses)
        num_domains = max(num_incoming_domains, num_internal_domains)

        if num_internal_domains < num_domains:
            # Expand the internal arrays to accommodate new domains
            size_diff = num_domains - num_internal_domains
            self.losses = np.concatenate([self.losses, np.zeros(size_diff, dtype=self.losses.dtype)])
            self.counts = np.concatenate([self.counts, np.zeros(size_diff, dtype=self.counts.dtype)])

        # Accumulate the incoming losses and counts
        self.losses[:num_incoming_domains] += losses
        self.counts[:num_incoming_domains] += counts

    def reset_state(self) -> None:
        """
        Resets the internal state (accumulated losses and counts).
        """
        self.losses = np.array([], dtype=self.losses.dtype)
        self.counts = np.array([], dtype=self.counts.dtype)

    @abstractmethod
    def calc_mixture(self) -> np.ndarray | None:
        """
        Computes the updated mixture coefficients based on the accumulated losses and counts.

        Subclasses must implement this method to define how the mixture is updated.

        Returns:
            A numpy array representing the new mixture coefficients, or None if no update is available.
        """
        raise NotImplementedError
