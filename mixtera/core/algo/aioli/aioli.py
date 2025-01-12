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
        aioli_normalize_a: bool = False,
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
            - aioli_normalize_a: whether or not to normalize the graph matrix before softmaxxing
            - aioli_diagonal: whether or not only considering self interactions
            - one_hot_factor: perturbation amount for the mixture coefficients
        """
        super().__init__()
        self.prior_steps = prior_steps
        self.eta = eta
        self.lp_rounds = lp_rounds
        self.lp_steps = lp_steps
        self.update_steps = update_steps
        self.aioli_normalize_a = aioli_normalize_a
        self.aioli_diagonal = aioli_diagonal
        self.domain_count = len(self.losses)
        self.graph = np.zeros((self.domain_count, self.domain_count))
        self.weights: np.ndarray | None = None  # The latest weights after update steps have been completed.
        self.one_hot_factor = one_hot_factor
        self.ema_graph = None
        self.ema = None

        self.last_generated_mixture = None  # The latest generated mixture weights. Can be perturbed or normal.

    def learn_params_subroutine(self) -> None:
        self.graph /= self.lp_rounds
        if self.one_hot_factor != 1 and not self.aioli_diagonal:
            weight_matrix = np.zeros((self.domain_count, self.domain_count))
            for i in range(self.domain_count):
                weight_row = np.ones(self.domain_count) * (1 - self.one_hot_factor) / (self.domain_count - 1)
                weight_row[i] = self.one_hot_factor
                # TODO(bguney): Vectorize the weight matrix
                weight_matrix[i] = weight_row

            new_graph = np.zeros((self.domain_count, self.domain_count))
            for i, row in enumerate(self.graph):
                new_graph[i] = np.linalg.solve(weight_matrix, row)

            self.graph = new_graph

        elif self.one_hot_factor != 1 and self.aioli_diagonal:
            new_graph = np.zeros((self.domain_count, self.domain_count))
            for i in range(self.domain_count):
                new_graph[i, i] = self.graph[i, i] / self.one_hot_factor

            self.graph = new_graph

    def process_losses(self, losses: np.ndarray, counts: np.ndarray, mixture_id: int) -> np.ndarray | None:
        update_at_client = False
        if mixture_id > self.last_received_mixture:
            update_at_client = True
            self.last_received_mixture = mixture_id

        if update_at_client:
            perturbed_domain = self.last_received_mixture % (self.domain_count + 1)
            if perturbed_domain != self.domain_count:
                self.graph[:, perturbed_domain] += self.losses - losses

        self._update_state(losses, counts)
        return self.calc_mixture(update_at_client)

    def _update_state(self, losses: np.ndarray, counts: np.ndarray) -> None:
        """
        Updates the losses and counts, adjusting internal arrays as needed to accommodate new domains.

        Args:
            losses: A numpy array of losses per domain.
            counts: A numpy array of counts per domain.
        """
        num_incoming_domains = len(losses)
        num_internal_domains = len(self.losses)
        num_domains = max(num_incoming_domains, num_internal_domains)

        super()._update_state(losses, counts)

        if num_internal_domains < num_domains:
            # Updating the relationship graph
            self.graph = np.zeros((num_domains, num_domains))

        # Assign the incoming losses and counts
        self.losses[:num_incoming_domains] = losses
        self.counts[:num_incoming_domains] = counts
        self.domain_count = len(self.losses)

    def calc_mixture(self, updated_at_client: bool) -> np.ndarray | None:
        if not updated_at_client:
            if self.weights is None:
                return self.initial_mixture
            return self.last_generated_mixture

        # Find which step we are in the algorithm given the mixture id.
        perturbed_domain = self.last_received_mixture % (self.domain_count + 1)

        # If we are out of the learn params phase, we compute the relationship graph.
        if perturbed_domain == self.domain_count:
            self.learn_params_subroutine()
            weights_init = np.ones(self.domain_count)

            logger.info(f"LearnParams done. New graph is {self.graph}")

            if self.aioli_normalize_a:
                min_entry = self.graph.min()
                if min_entry < 0:
                    self.graph -= min_entry
                    logger.info(f"Rescaled graph is {self.graph}, previous min entry was {min_entry}")

                    self.graph /= self.graph.sum()
                    logger.info(f"Graph after normalization: {self.graph}")

            if self.ema is not None:
                self.ema_graph = (
                    self.graph if self.ema_graph is None else (1 - self.ema) * self.graph + self.ema * self.ema_graph
                )
                logger.info(f"Applying ema, smoothed graph is {self.ema_graph}")
                self.weights = np.multiply(weights_init, np.exp(self.eta * self.ema_graph.sum(axis=0)))
            else:
                weights = weights_init if self.weights is None else self.weights
                self.weights = np.multiply(weights, np.exp(self.eta * self.graph.sum(axis=0)))

            logger.info(f"The new mixture proportions={self.weights/sum(self.weights)}. ")
            self.weights = self.weights / np.sum(self.weights)
            self.last_generated_mixture = self.weights
            return self.weights

        self.last_generated_mixture = np.ones(self.domain_count) * (1 - self.one_hot_factor) / (self.domain_count - 1)
        self.last_generated_mixture[perturbed_domain] = self.one_hot_factor
        return self.last_generated_mixture
