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
        ema: float | None = None,
        lp_duration: int = 1000,
        lp_rounds: int = 1,
        lp_sweep: int = 1,
        aioli_normalize_a: bool = False,
        aioli_diagonal: bool = False,
        one_hot_factor: float = 1,
        prior_steps: int = -1,
        lp_portion: float = 0.1,
    ):
        """
        Initializes the Aioli dynamic mixing algorithm.
        Args:
            - prior_steps: how many steps to run the prior (to simulate this transfer setting)
            - eta: softmax temperature hyperparameter
            - lp_rounds: number of sweeps through the k dataset
            - lp_duration: duration of the learn params
            - lp_portion: perturbation portion in the learn params phase
            - aioli_normalize_a: whether or not to normalize the graph matrix before softmaxxing
            - aioli_diagonal: whether or not only considering self interactions
            - one_hot_factor: perturbation amount for the mixture coefficients
        """
        super().__init__()
        self.prior_steps = prior_steps
        self.eta = eta
        self.lp_rounds = lp_rounds
        self.lp_duration = lp_duration
        self.lp_portion = lp_portion
        self.lp_sweep = lp_sweep
        self.aioli_normalize_a = aioli_normalize_a
        self.aioli_diagonal = aioli_diagonal
        self.graph: np.ndarray | None = None
        self.weights: np.ndarray | None = None  # The latest weights after update steps have been completed.
        self.one_hot_factor = one_hot_factor
        self.ema_graph = None
        self.ema = ema

        self.total_steps = 0

    def learn_params_subroutine(self) -> None:
        dynamic_steps = self.total_steps - self.prior_steps
        current_round = (dynamic_steps + self.lp_duration - 1) // self.lp_duration

        self.graph /= current_round + 1
        domain_count = len(self.counts)

        if self.one_hot_factor != 1 and not self.aioli_diagonal:
            weight_matrix = np.zeros((domain_count, domain_count))
            for i in range(domain_count):
                weight_row = np.ones(domain_count) * (1 - self.one_hot_factor) / (domain_count - 1)
                weight_row[i] = self.one_hot_factor
                # TODO(bguney): Vectorize the weight matrix
                weight_matrix[i] = weight_row

            new_graph = np.zeros((domain_count, domain_count))
            for i, row in enumerate(self.graph):
                new_graph[i] = np.linalg.solve(weight_matrix, row)

            self.graph = new_graph

        elif self.one_hot_factor != 1 and self.aioli_diagonal:
            new_graph = np.zeros((domain_count, domain_count))
            for i in range(domain_count):
                new_graph[i, i] = self.graph[i, i] / self.one_hot_factor

            self.graph = new_graph

    def _update_state(self, losses: np.ndarray, counts: np.ndarray) -> None:
        """
        Updates the losses and counts, adjusting internal arrays as needed to accommodate new domains.

        Args:
            losses: A numpy array of losses per domain.
            counts: A numpy array of counts per domain.
        """
        self.total_steps += 1
        num_incoming_domains = len(losses)
        num_internal_domains = len(self.losses)
        num_domains = max(num_incoming_domains, num_internal_domains)

        if num_internal_domains < num_domains:
            # Updating the relationship graph to accommadate the new domains
            self.graph = np.zeros((num_domains, num_domains))
        else:
            # Adding the loss to the graph
            if self.total_steps > self.prior_steps:
                if self.is_in_perturbation() and self.perturbed_domain < num_domains:
                    self.graph[:, self.perturbed_domain] += self.losses - losses

        if num_internal_domains < num_domains:
            # Expand the internal arrays to accommodate new domains
            size_diff = num_domains - num_internal_domains
            self.losses = np.concatenate([self.losses, np.zeros(size_diff, dtype=self.losses.dtype)])
            self.counts = np.concatenate([self.counts, np.zeros(size_diff, dtype=self.counts.dtype)])

        # Assign the incoming losses and counts
        self.losses[:num_incoming_domains] = losses
        self.counts[:num_incoming_domains] = counts

    @property
    def perturbed_domain(self) -> int:
        """
        Method to return the currently perturbed index
        """
        dynamic_steps = self.total_steps - self.prior_steps
        proportion = int(self.lp_duration * self.lp_portion)
        domain_count = len(self.losses)
        steps_per_perturbation = proportion / (max(domain_count, 1) * self.lp_sweep)
        location_within_phase = dynamic_steps % self.lp_duration
        perturbed_domain = int(location_within_phase // steps_per_perturbation)
        return perturbed_domain % domain_count

    def is_in_perturbation(self) -> bool:
        """
        Checking if the model is being trained on perturbed domains.
        """
        dynamic_steps = self.total_steps - self.prior_steps
        initial_domain = dynamic_steps % self.lp_duration
        proportion = int(self.lp_duration * self.lp_portion)
        return initial_domain < proportion

    def calc_mixture(self, updated_at_client: bool) -> np.ndarray | None:
        # If we are in the prior steps, we return the initial mixture.
        if self.total_steps < self.prior_steps:
            return None

        dynamic_steps = self.total_steps - max(self.prior_steps, 0)
        current_round = (dynamic_steps + self.lp_duration - 1) // self.lp_duration
        proportion = int(self.lp_duration * self.lp_portion)
        domain_count = len(self.losses)
        location_within_phase = dynamic_steps % self.lp_duration

        if current_round > self.lp_rounds:
            logger.info("All the learn params rounds have been executed.")
            return None

        if proportion < domain_count:
            logger.error("There is not enough steps to pertub all the domains")
            return None

        # If we are out of the perturbation phase, we train with the newly learned weights
        if location_within_phase > proportion:
            return None

        # If we are out of the learn params phase, we compute the relationship graph.
        if location_within_phase == proportion:
            self.learn_params_subroutine()
            weights_init = np.ones(domain_count)

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
            logger.info(f"Sum of weights sum{sum(self.weights)}, ***** {self.weights}")
            self.weights = self.weights / np.sum(self.weights)
            return self.weights

        last_generated_mixture = np.ones(domain_count) * (1 - self.one_hot_factor) / (domain_count - 1)
        last_generated_mixture[self.perturbed_domain] = self.one_hot_factor
        return last_generated_mixture
