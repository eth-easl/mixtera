import numpy as np
from loguru import logger
from mixtera.core.algo.dynamic_mixing.dynamic_mixing import DynamicMixingAlgorithm


class AioliDynamicMixing(DynamicMixingAlgorithm):
    """
    Aioli dynamic mixing algorithm implementation.
    This class implements the Aioli algorithm for dynamically adjusting mixture coefficients
    based on domain interactions deducted from the accumulated losses and counts.
    """

    def __init__(
        self,
        eta: float = 0.1,
        lp_steps: int = 10,
        lp_rounds: int = 1,
        update_steps: int = 1000,
        aioli_normalize_A: bool = False,
        aioli_diagonal: bool = False,
        one_hot_factor: float = 1,
        prior_steps: int = -1,
    ):
        """
        Initializes the Aioli dynamic mixing algorithm.
        Args:
            - prior_steps: how many steps to run the prior (to simulate this transfer setting)
            - eta: softmax temperature hyperparameter
            - lp_rounds: number of sweeps through the k dataset
            - lp_steps: number of contiguous batches to take for each dataset
            - update_steps: how many steps to update weights
            - aioli_normalize_A: whether or not to normalize the graph matrix before softmaxxing
            - aioli_diagonal: whether or not only considering self interactions
            - one_hot_factor: perturbation amount for the mixture coefficients
        """
        super().__init__()
        self.prior_steps = prior_steps
        self.eta = eta
        self.lp_rounds = lp_rounds
        self.lp_steps = lp_steps
        self.update_steps = update_steps
        self.aioli_normalize_A = aioli_normalize_A
        self.aioli_diagonal = aioli_diagonal
        self.domain_count = len(self.losses)
        self.graph = np.zeros((self.domain_count, self.domain_count))
        self.previous_loss = np.array([], dtype=np.float32)
        self.weights = None
        self.one_hot_factor = one_hot_factor
        self.ema_graph = None
        self.ema = None

    @property
    def current_perturbed_domain(self, training_steps: int) -> int:
        lp_duration = self.lp_rounds * self.lp_steps * self.domain_count
        remaining = training_steps % self.update_steps

        if remaining <= lp_duration:
            return remaining / self.lp_steps
        else:
            logger.info("Currently not in the learn parameters phase.")
            return -1

    def process_losses(
        self, losses: np.ndarray, counts: np.ndarray, training_steps: int | None = None
    ) -> np.ndarray | None:
        self._update_state(losses, counts)

        perturbed_domain = self.current_perturbed_domain(training_steps)
        if perturbed_domain != -1:
            self.graph[:, perturbed_domain] += self.previous_loss - losses
            self.previous_loss = losses
        return self.calc_mixture(training_steps)

    def learn_params_subroutine(self) -> None:
        self.graph /= self.lp_rounds
        if self.one_hot_factor != 1 and not self.aioli_diagonal:
            weight_matrix = np.zeros((self.domain_count, self.domain_count))
            for i in range(self.domain_count):
                weight_row = np.ones(self.domain_count) * (1 - self.one_hot_factor) / (self.domain_count - 1)
                weight_row[i] = self.one_hot_factor
                weight_matrix[i] = weight_row

            A = np.zeros((self.domain_count, self.domain_count))
            for i, row in enumerate(self.graph):
                A[i] = np.linalg.solve(weight_matrix, row)

            self.graph = A

        elif self.one_hot_factor != 1 and self.aioli_diagonal:
            A = np.zeros((self.domain_count, self.domain_count))
            for i in range(self.domain_count):
                A[i, i] = self.graph[i, i] / self.one_hot_factor

            self.graph = A

    def calc_mixture(self, training_steps: int) -> np.ndarray | None:
        """
        Computes the updated mixture coefficients based on the accumulated losses and counts.

        Args:
            training_steps: The current training steps of the model.

        Returns:
            A numpy array representing the new mixture coefficients, or None if no update is available.
        """
        if self.prior_steps != -1 and self.prior_steps >= training_steps:
            self.weights = self.initial_mixture
            return self.weights

        if self.current_perturbed_domain(training_steps) != -1:
            self.weights = np.ones(self.domain_count) * (1 - self.one_hot_factor) / (self.domain_count - 1)
            self.weights[self.current_perturbed_domain] = self.one_hot_factor
            return self.weights

        # If we are out of the learn params phase, we compute the relationship graph.
        self.learn_params_subroutine()
        weights_init = np.ones(self.domain_count)

        self.logger.info(f"LearnParams done. New graph is {self.graph}")

        if self.aioli_normalize_A:
            min_entry = self.graph.min()
            if min_entry < 0:
                self.graph -= min_entry
                self.logger.info(f"Rescaled graph is {self.graph}, previous min entry was {min_entry}")

                self.graph /= self.graph.sum()
                self.logger.info(f"Graph after normalization: {self.graph}")

        if self.ema is not None:
            if self.ema_graph is None:
                self.ema_graph = self.graph
            else:
                self.ema_graph = (1 - self.ema) * self.graph + self.ema * self.ema_graph
            self.logger.info(f"Applying ema, smoothed graph is {self.ema_graph}")
            self.weights = np.multiply(weights_init, np.exp(self.eta * self.ema_graph.sum(axis=0)))
        else:
            if self.weights is None:
                self.weights = np.multiply(weights_init, np.exp(self.eta * self.graph.sum(axis=0)))
            else:
                self.weights = np.multiply(self.weights, np.exp(self.eta * self.graph.sum(axis=0)))

        self.logger.info(f"The new mixture proportions={self.weights/sum(self.weights)}. ")
        return self.weights / sum(self.weights)
