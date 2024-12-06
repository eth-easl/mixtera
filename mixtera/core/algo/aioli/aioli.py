import json

import numpy as np
from loguru import logger
from mixtera.core.algo.dynamic_mixing.dynamic_mixing import DynamicMixingAlgorithm
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.special import logsumexp
from tqdm import tqdm


class AioliDynamicMixing(DynamicMixingAlgorithm):
    """
    Aioli dynamic mixing algorithm implementation.
    This class implements the ADO algorithm for dynamically adjusting mixture coefficients
    based on domain interactions deducted from the accumulated losses and counts.
    """

    def __init__(self, prior_steps: int, eta: float, lp_rounds: int, lp_steps: int, update_steps: int, aioli_normalize_A: bool, aioli_diagonal: bool, one_hot_factor: float):
        """
        Initializes the Aioli dynamic mixing algorithm.
        Args:
            - prior_steps: how many steps to run the prior (to simulate this transfer setting)
            - eta: softmax temperature hyperparameter 
            - lp_rounds: number of sweeps through the k dataset 
            - lp_steps: number of contiguous batches to take for each dataset
            - update_steps: how many steps to update weights
            - aioli_normalize_A: whether or not to normalize the graph matrix before softmaxxing
        """
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
        self.one_hot_factor = one_hot_factor
        self.ema_graph = None
        self.ema = None
        self.all_graphs = []

    def _in_learn_params_phase(self, training_steps: int) -> bool:
        assert training_steps >= 0, "The training steps of the model should be greater than 0."
        lp_duration = self.lp_rounds * self.lp_steps * self.domain_count
        remaining = training_steps % self.update_steps
        return remaining <= lp_duration

    # Returns the index of the currently perturbed domain
    @property
    def current_perturbed_domain(self, training_steps: int) -> int:
        lp_duration = self.lp_rounds * self.lp_steps * self.domain_count
        remaining = training_steps % self.update_steps

        if remaining <= lp_duration:
            return remaining / self.lp_steps
        else:
            logger.info("Currently not in the learn parameters phase.")
            return -1
        
    def process_losses(self, losses: np.ndarray, counts: np.ndarray, training_steps: int | None = None) -> np.ndarray | None:
        self._update_state(losses, counts)

        perturbed_domain = self.current_perturbed_domain(training_steps)
        if perturbed_domain != -1:
            self.graph[:, perturbed_domain] += self.previous_loss - losses
            self.previous_loss = losses
        return self.calc_mixture(False, training_steps)

    def learn_params_subroutine(self) -> None:
        self.graph /= self.lp_rounds
        if self.one_hot_factor != 1 and not self.aioli_diagonal:
            #print(f"New graph before: {new_graph}")
            weight_matrix = np.zeros((self.domain_count, self.domain_count))
            for i in range(self.domain_count):
                weight_row = np.ones(self.domain_count) * (1 - self.one_hot_factor)/(self.domain_count - 1)
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

    def calc_mixture(self, updated_at_client: bool, training_steps: int) -> np.ndarray | None:
        if self.current_perturbed_domain != -1:
            perturbed_weights = np.ones(self.domain_count) * (1 - self.one_hot_factor)/(self.domain_count-1)
            perturbed_weights[self.current_perturbed_domain] = self.one_hot_factor
            return perturbed_weights
        
        self.learn_params_subroutine()
        self.all_graphs.append(self.graph)
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
            if self.ema_graph == None:
                self.ema_graph = self.graph
            else:
                self.ema_graph = (1-self.ema) * self.graph + self.ema * self.ema_graph
            self.logger.info(f"Applying ema, smoothed graph is {self.ema_graph}")
            weights = np.multiply(weights_init, np.exp(self.eta * self.ema_graph.sum(axis=0)))
        else:
            if i == 0:
                weights = np.multiply(weights_init, np.exp(self.args.eta * self.graph.sum(axis=0)))
            else:
                weights = np.multiply(weights, np.exp(self.args.eta * self.graph.sum(axis=0)))

            self.logger.info(f"new data distribution={weights/sum(weights)}. ")
            return weights / sum(weights)
    