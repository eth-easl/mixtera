# pylint: disable=invalid-name
# Mathematical notation with snake-case is harder to read.

import json
from typing import Any

import numpy as np
from loguru import logger
from mixtera.core.algo.dynamic_mixing.dynamic_mixing import DynamicMixingAlgorithm
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.special import logsumexp
from tqdm import tqdm


class AdoDynamicMixing(DynamicMixingAlgorithm):
    """
    Adaptive Data Optimization (ADO) dynamic mixing algorithm implementation.

    This class implements the ADO algorithm for dynamically adjusting mixture coefficients
    based on domain-specific scaling laws fitted to the accumulated losses and counts.

    There currently are 2 known differences to the paper:
        1) We use a different  clip_min_probability implementation and clip pi_t,
            while the paper implementation clips rho_t.
        2) We normalize rho_t and pi_bar_t to be a true distribution after weighting
            its components, but the paper implementation does not normalize.
    """

    def __init__(
        self,
        variant: str = "vanilla",
        gamma1: float = 0.1,
        gamma2: float = 0.1,
        s: float = 0.5,
        delta_min: float = 0.01,
        scaling_law_update_interval: int = 1000,
        subsampling_interval: int = 10,
        ignore_initial_steps: int = 500,
        savgol: bool = True,
        use_same_step_size: bool = True,
        count_normalizer: None | int = None,
        logging_path: str | None = None,
    ) -> None:
        """
        Initializes the ADO dynamic mixing algorithm.

        Args:
            variant: The variant of the algorithm ('vanilla', 'adjusted_v1', 'adjusted_v2').
            gamma1: The smoothing factor for the credit assignment score h(t).
            gamma2: The smoothing factor for the data policy pi(t).
            s: The exponent used in computing the credit assignment score lambda_k(t).
            delta_min: The minimum probability assigned to each domain that has been sampled at least once.
            scaling_law_update_interval: The number of steps between each scaling law parameter update.
            subsampling_interval: For subsampling during scaling law fitting
            ignore_initial_steps: Number of initial steps to ignore before fitting scaling laws.
            savgol: Whether to apply a savgol filter on the losses to smoothen them
            use_same_step_size: If True, all sampled domains increase their count. Done in the original paper
                                to more scaling laws more comparable and account for transfer learning.
            count_normalizer: If set to a positive int, we divide all counts by this when interacting with scaling laws.
                              Can be helpful since the bounds we currently use
                              for parameters are provided by the original paper.
                              They used sample count with 1024 tokens/sample. Hence, to align with the paper, _if_
                              your inputs are tokens (default assumption by Mixtera), this needs to be 1024.
            logging_path: If provided, we will save debug json logs to this path.
        """
        super().__init__()
        self.variant = variant
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.s = s
        self.delta_min = delta_min
        self.scaling_law_update_interval = scaling_law_update_interval
        self.ignore_initial_steps = ignore_initial_steps
        self.subsampling_interval = subsampling_interval
        self.savgol = savgol
        self.logging_path = logging_path
        self.use_same_step_size = use_same_step_size
        self.count_normalizer = count_normalizer

        # Initialize per-domain data structures
        self.total_steps = 0  # Total number of steps processed
        self.last_update_step = 0  # Step when we last updated h_t

        # Initialize h(t), pi(t), and moving averages
        self.h_t: np.ndarray | None = None  # Credit assignment score h_k(t)
        self.pi_t: np.ndarray | None = None  # Data policy pi_k(t)
        self.pi_bar_t_minus_1: np.ndarray | None = None  # Previous moving average pī_k(t-1)
        self.scaling_law_params: np.ndarray | None = None  # Scaling law parameters [log_beta_k, log_epsilon_k, alpha_k]

        # Keep track of per-step counts and losses
        self.per_step_counts: list[np.ndarray] = []  # List of counts per step per domain
        self.per_step_losses: list[np.ndarray] = []  # List of losses per step per domain

        # Initial prior distribution mu_k (from initial_mixture)
        self.mu_k: np.ndarray | None = None  # Will be set based on self.initial_mixture

        self.next_continue_at: int | None = None
        self.handed_out_first_update = False

        self.log_counts: list[np.ndarray] | None = None
        if self.logging_path is not None:
            self.log_config = {
                "variant": variant,
                "gamma1": gamma1,
                "gamma2": gamma2,
                "s": s,
                "delta_min": delta_min,
                "scaling_law_update_interval": scaling_law_update_interval,
                "subsampling_interval": subsampling_interval,
                "ignore_initial_steps": ignore_initial_steps,
                "savgol": savgol,
            }
            self.log_entries: list[Any] = []
            self.log_scaling_laws: list[Any] = []
            self.log_counts = []

    def __getstate__(self) -> dict[Any, Any]:
        state = self.__dict__.copy()
        if "log_entries" in state:
            state.pop("log_entries", None)
        if "log_scaling_laws" in state:
            state.pop("log_scaling_laws", None)
        if "log_counts" in state:
            state.pop("log_counts", None)
        return state

    def __setstate__(self, state: dict[Any, Any]) -> None:
        # Restore the object's state
        self.__dict__.update(state)
        # Reinitialize the logging attributes to their initial state
        self.log_entries = []
        self.log_scaling_laws = []
        self.log_counts = [] if self.logging_path is not None else None

    def __str__(self) -> str:
        """
        Returns a dictionary-like string representation of the AdoDynamicMixing's current state.
        """
        state_dict = {
            "variant": self.variant,
            "total_steps": self.total_steps,
            "gamma1": self.gamma1,
            "gamma2": self.gamma2,
            "s": self.s,
            "delta_min": self.delta_min,
            "scaling_law_update_interval": self.scaling_law_update_interval,
            "ignore_initial_steps": self.ignore_initial_steps,
            "subsampling_interval": self.subsampling_interval,
            "num_domains": len(self.counts) if self.counts is not None else None,
        }

        if self.mu_k is not None:
            state_dict["mu_k"] = self.mu_k.tolist()
        else:
            state_dict["mu_k"] = None

        if self.pi_t is not None:
            state_dict["pi_t"] = self.pi_t.tolist()
        else:
            state_dict["pi_t"] = None

        if self.h_t is not None:
            state_dict["h_t"] = self.h_t.tolist()
        else:
            state_dict["h_t"] = None

        if self.scaling_law_params is not None:
            # Convert scaling_law_params to list of dictionaries for readability
            scaling_params_list = []
            for idx, params in enumerate(self.scaling_law_params):
                log_beta_k, log_epsilon_k, alpha_k = params
                scaling_params_list.append(
                    {
                        "domain": idx,
                        "log_beta_k": log_beta_k,
                        "log_epsilon_k": log_epsilon_k,
                        "alpha_k": alpha_k,
                    }
                )
            state_dict["scaling_law_params"] = scaling_params_list
        else:
            state_dict["scaling_law_params"] = None

        if self.counts is not None:
            state_dict["counts"] = self.counts.tolist()
        else:
            state_dict["counts"] = None

        if self.losses is not None:
            state_dict["losses"] = self.losses.tolist()
        else:
            state_dict["losses"] = None

        return str(state_dict)

    def write_logs(self) -> None:
        if self.logging_path is not None:
            log_data = {
                "config": self.log_config,
                "entries": self.log_entries,
                "scaling_laws": self.log_scaling_laws,
                "log_counts": self.log_counts,
                "mu_k": self.mu_k.tolist() if self.mu_k is not None else None,
            }
            with open(self.logging_path, "w+", encoding="utf-8") as f:
                json.dump(log_data, f, indent=4)

    def calc_mixture(self, updated_at_client: bool) -> np.ndarray | None:
        """
        Computes the updated mixture coefficients based on the accumulated losses and counts.

        Args:
            updated_at_client: A boolean flag indicating whether a new mixture has been received
                            at the client (used for handling mixture update delays).

        Returns:
            A numpy array representing the new mixture coefficients, or None if no update is available.
        """
        self.total_steps += 1

        num_domains = len(self.counts)

        # Initialize mu_k if not set
        if self.mu_k is None:
            # Use the initial mixture as the prior
            self.mu_k = self.initial_mixture.copy()
            # Might need to adjust size
            if (size_diff := num_domains - len(self.mu_k)) > 0:
                self.mu_k = np.concatenate([self.mu_k, np.zeros(size_diff, dtype=self.mu_k.dtype)])

            assert size_diff >= 0, f"size_diff = {size_diff}"
            assert self.mu_k is not None
            assert np.isclose(1, np.sum(self.mu_k))
            assert (
                len(self.mu_k) == num_domains
            ), f"len(self.mu_k) = {len(self.mu_k)} != num_domains = {num_domains}\n\n{self.mu_k}"

        # **Warm-up Handling**
        if self.total_steps <= self.ignore_initial_steps + self.scaling_law_update_interval:
            # During warm-up, we use the initial mixture and do not update it
            # Since we cannot use the first information during self.ignore_initial_steps,
            # we need to wait an additional interval before actually starting the update.
            if self.logging_path is not None:
                # Still write log information if necessary.
                step_log = {
                    "step": self.total_steps,
                    "counts": self.counts.tolist() if self.counts is not None else None,
                    "losses": self.losses.tolist() if self.losses is not None else None,
                }
                self.log_entries.append(step_log)
                if self.total_steps % 50 == 0:
                    self.write_logs()
            return None

        # Initialize h_t if not set
        self.h_t = self.h_t if self.h_t is not None else self.mu_k.copy()

        updated_scaling_laws = False
        # Fit scaling laws immediately after the warm-up period ends, and then at intervals
        if (self.total_steps == self.ignore_initial_steps + 1) or (
            self.total_steps > self.ignore_initial_steps
            and (self.total_steps - self.ignore_initial_steps - 1) % self.scaling_law_update_interval == 0
        ):
            self.fit_scaling_laws()
            updated_scaling_laws = True

        # Only relevant for adjusted_v3
        should_continue = True
        force_log = False
        if self.variant == "adjusted_v3":
            if self.handed_out_first_update:
                should_continue = False
                if updated_at_client:
                    self.next_continue_at = self.total_steps + 15  # give us some slack to collect more data.
                    logger.debug(f"Updated at client, continuting at {self.next_continue_at}")

                if (
                    self.next_continue_at is not None and self.total_steps == self.next_continue_at
                ) or updated_scaling_laws:
                    # We also get a new mixture ID if the chunk-level mixture does not change,
                    # and updated_scaling_laws fixes that.
                    logger.debug("Continuing!")
                    should_continue = True
                    force_log = True

        if not should_continue:
            return None
        # End code for adjusted_v3

        assert self.scaling_law_params is not None

        # Compute the derivative of loss with respect to n for each domain
        dL_dn = self.compute_loss_derivative()

        # Compute the preference distribution rho_k(t)
        self.compute_rho_t(dL_dn)

        # Update the data policy pi_k(t)
        self.update_pi_t()

        # Enforce minimum probability delta_min for domains that have been sampled at least once
        self.enforce_min_probability()

        # Update h(t) based on the variant and whether the mixture was updated at the client
        if self.variant in ("vanilla", "adjusted_v3"):
            # Update h(t) every step using the last calculated pi(t)
            self.update_h_t()
        elif self.variant == "adjusted_v1":
            # h(t) remains identical until we receive an update based on a new mixture_id
            if updated_at_client:
                self.update_h_t()
        elif self.variant == "adjusted_v2":
            # Similar to adjusted_v1, but adjust the moving average to not let h(t-1) dominate
            if updated_at_client:
                steps_elapsed = self.total_steps - self.last_update_step
                self.update_h_t(elapsed_steps=steps_elapsed)
        else:
            raise ValueError(f"Unknown variant '{self.variant}' specified.")

        # Update pī_k(t)
        self.update_pi_bar_t()

        logger.debug(f"Calculated mixture = {self.pi_t}")

        if self.logging_path is not None:
            step_log = {
                "step": self.total_steps,
                "counts": self.counts.tolist() if self.counts is not None else None,
                "losses": self.losses.tolist() if self.losses is not None else None,
                "h_t": self.h_t.tolist() if self.h_t is not None else None,
                "pi_t": self.pi_t.tolist() if self.pi_t is not None else None,
                "pi_bar_t_minus_1": self.pi_bar_t_minus_1.tolist() if self.pi_bar_t_minus_1 is not None else None,
                "rho_t": self.rho_t.tolist() if hasattr(self, "rho_t") else None,
                "dL_dn": dL_dn.tolist() if "dL_dn" in locals() else None,
                "scaling_law_params": self.scaling_law_params.tolist() if self.scaling_law_params is not None else None,
            }
            self.log_entries.append(step_log)

        self.handed_out_first_update = True

        if (self.total_steps % 50 == 0) or force_log:
            self.write_logs()

        return self.pi_t

    def update_h_t(self, elapsed_steps: int = 1) -> None:
        """
        Updates the credit assignment score h_k(t) based on the data policy pi_k(t).

        Args:
            elapsed_steps: Number of steps elapsed since the last update (used in variant 'adjusted_v2').
        """
        assert self.mu_k is not None and self.h_t is not None and self.pi_t is not None

        gamma1 = self.gamma1

        if self.variant == "adjusted_v2" and elapsed_steps > 1:
            # Adjust gamma1 to account for the fact that h(t-1) remained constant over elapsed_steps
            gamma1 = 1 - (1 - gamma1) ** elapsed_steps

        self.h_t = gamma1 * self.pi_t + (1 - gamma1) * self.h_t
        self.last_update_step = self.total_steps

    def fit_scaling_laws(self) -> None:
        """
        Fits scaling laws to the accumulated counts and losses for each domain.
        """
        logger.debug("Re-fitting scaling laws.")
        num_domains = len(self.counts)
        self.scaling_law_params = np.zeros((num_domains, 3))  # [log_beta_k, log_epsilon_k, alpha_k]
        assert self.scaling_law_params is not None

        counts_over_time = np.array(self.per_step_counts)
        losses_over_time = np.array(self.per_step_losses)

        # Compute cumulative counts per domain
        cumulative_counts = np.cumsum(counts_over_time, axis=0)

        # TODO(#create issue): We could use a multiprocessing Pool here.
        for k in tqdm(range(num_domains), desc="Fitting per-domain scaling laws.", total=num_domains, unit="domains"):
            nonc_counts_k = counts_over_time[:, k]  # Non cumulative counts over time
            counts_k = cumulative_counts[:, k]
            losses_k = losses_over_time[:, k]
            steps_k = np.arange(len(counts_k))

            # First, we need to clean up the y data (losses)
            # We either impute values (as per the official implementation)
            # or remove them.

            if self.use_same_step_size:
                # When using the same step size, we impute missing losses
                # This aligns with the paper but we could also make this a flag at some point
                # to either impute losses or select only losses > 0.
                for t in range(1, len(losses_k)):
                    if losses_k[t] == 0:
                        losses_k[t] = losses_k[t - 1]
            else:
                # If fitting on x data for each domain, select only items where loses are > 0
                valid_indices = losses_k > 0
                counts_k = counts_k[valid_indices]
                losses_k = losses_k[valid_indices]
                steps_k = steps_k[valid_indices]
                nonc_counts_k = nonc_counts_k[valid_indices]

            # After having a continous set of losses, we optionally apply a savgol filter
            applied_savgol = False
            if self.savgol:
                window_length = min(101, len(counts_k))
                if window_length % 2 == 0:
                    window_length -= 1  # window_length must be odd
                    logger.debug(f"Adjusted window length to {window_length} for domain {k}")
                if window_length > 3:
                    losses_k = savgol_filter(losses_k.copy(), window_length=window_length, polyorder=3)
                    applied_savgol = True
                else:
                    logger.debug(f"Not enough data points to apply Savitzky-Golay filter for domain {k}.")

            # As a third step, we only select entries in our time series where we sampled data up to that point.
            # At the beginning, we might have domains that we did not sample yet, and there we cannot run any prediction
            # Hence, this filters out those points at the start.
            valid_indices = counts_k > 0
            counts_k = counts_k[valid_indices]
            losses_k = losses_k[valid_indices]
            steps_k = steps_k[valid_indices]
            nonc_counts_k = nonc_counts_k[valid_indices]

            # Last, we filter out losses that were collected before the ignore_initial_steps option.
            valid_indices = steps_k > self.ignore_initial_steps
            counts_k = counts_k[valid_indices]
            losses_k = losses_k[valid_indices]
            steps_k = steps_k[valid_indices]
            nonc_counts_k = nonc_counts_k[valid_indices]

            # Subsample data
            subsampled = False
            if self.subsampling_interval > 1:
                counts_k = counts_k[:: self.subsampling_interval]
                losses_k = losses_k[:: self.subsampling_interval]
                steps_k = steps_k[:: self.subsampling_interval]
                subsampled = True

            if len(counts_k) < 1:
                logger.debug(
                    f"Too little data to fit scaling laws for domain {k}. Most likely due to unsampled domain."
                )
                if self.counts[k] > 0:
                    raise RuntimeError(
                        f"We sampled domain {k}"
                        + "during the initial steps but not between that and fitting the first scaling laws."
                        + "Please either increase your interval size or"
                        + "decrease or warm up steps to ensure this does not happen."
                    )

                self.scaling_law_params[k] = (-1, -1, -1)
                continue

            x_data = counts_k
            y_data = losses_k

            if self.count_normalizer is not None and self.count_normalizer > 1:
                # Can be used to convert tokens into samples
                x_data = x_data / float(self.count_normalizer)

            # **Define the grid of initializations as per the paper**
            alpha_grid = np.array([0.1 * i for i in range(1, 8)])
            log_beta_grid = np.array(list(range(-2, 6)))
            # Note that the paper enforces log epsilon > 0.5 but nevertheless uses this as the init grid.
            # To get the same results as the paper, we use this grid.
            log_epsilon_grid = np.array([-2.0, -1.5, -1.0, -0.5, 1.0, 1.5])

            # Create all combinations of initial guesses
            grid_search = [
                (log_beta_0, log_epsilon_0, alpha_0)
                for alpha_0 in alpha_grid
                for log_beta_0 in log_beta_grid
                for log_epsilon_0 in log_epsilon_grid
            ]

            best_loss = np.inf
            best_params = None

            for initial_guess in grid_search:
                # We don't use the bounds parameter since this gives different results than the paper.
                # Instead, as in the original implementation, we rely on penalty terms in the loss.
                result = minimize(
                    AdoDynamicMixing.scaling_law_loss,
                    initial_guess,
                    args=(x_data, y_data),
                    method="L-BFGS-B",
                    options={"maxiter": 200, "gtol": 1e-5},
                )

                if result.success and result.fun < best_loss:
                    # logger.debug(f"Found new params = {result.x} with loss {result.fun} < {best_loss}")
                    best_loss = result.fun
                    best_params = result.x

            if best_params is not None:
                assert self.scaling_law_params is not None
                self.scaling_law_params[k] = best_params
                logger.debug(f"Selected best_params {best_params} with loss = {best_loss}")
            else:
                # Handle optimization failure (e.g., keep previous parameters or use default)
                raise RuntimeError(f"Error while fitting scaling law!\n{result}")

            if self.logging_path is not None:
                domain_log = {
                    "step": self.total_steps,
                    "domain_index": k,
                    "counts_k": counts_k.tolist(),
                    "losses_k": losses_k.tolist(),
                    "x_data": x_data.tolist(),
                    "y_data": y_data.tolist(),
                    "steps_k": steps_k.tolist(),
                    "best_params": best_params.tolist() if best_params is not None else None,
                    "window_length": window_length if "window_length" in locals() else None,
                    "best_loss": best_loss,
                    "applied_savgol": applied_savgol,
                    "subsampled": subsampled,
                }
                self.log_scaling_laws.append(domain_log)

        logger.debug("Finished fitting scaling laws.")

    @staticmethod
    def scaling_law_loss(params: tuple[float, float, float], counts_k: int, losses_k: float) -> float | np.floating:
        # Fit the scaling law: L_k(n) = ε_k + β_k * n^{-α_k}
        # We fit in log-space to stabilize the optimization
        log_beta_k, log_epsilon_k, alpha_k = params

        # Check for invalid parameter values
        if not np.isfinite(log_beta_k) or not np.isfinite(log_epsilon_k) or not np.isfinite(alpha_k):
            return np.inf

        beta_k = np.exp(log_beta_k)
        epsilon_k = np.exp(log_epsilon_k)
        pred = logsumexp([log_beta_k - alpha_k * np.log(counts_k), log_epsilon_k + np.zeros_like(counts_k)], axis=0)
        target = np.log(losses_k)

        if np.isnan(beta_k):
            raise RuntimeError(f"beta_k is nan. log_beta_k = {log_beta_k} params = {params}")

        if np.isnan(epsilon_k):
            raise RuntimeError(f"epsilon_k is nan. log_epsilon_k = {log_epsilon_k} params = {params}")

        if any(np.isnan(pred)):
            raise RuntimeError(
                f"pred has nan = {pred} is nan = {np.isnan(pred)}. epsilon_k = {epsilon_k}, "
                + f"beta_k = {beta_k} counts_k = {counts_k} alpha_k = {alpha_k} params = {params}"
            )

        if any(np.isnan(target)):
            raise RuntimeError(
                f"target has nan = {target} is nan = {np.isnan(target)}. losses_k = {losses_k} params = {params}"
            )

        # Huber loss for robustness
        delta = 1e-3
        abs_diff = np.abs(pred - target)
        squared_loss = np.where(abs_diff <= delta, 0.5 * abs_diff**2, delta * (abs_diff - 0.5 * delta))
        # Constraints to keep parameters in reasonable ranges
        penalty = 0.0
        penalty += max(0, alpha_k - 0.8) * 1e3  # α_k <= 0.8
        penalty += max(0, 0.001 - alpha_k) * 1e3  # α_k >= 0 roughly (as in the original implementation)
        penalty += max(0, log_beta_k - 6.5) * 1e3  # log_beta_k <= 6.5
        penalty += max(0, 0.5 - log_epsilon_k) * 1e3  # log_epsilon_k >= 0.5
        loss = np.mean(squared_loss) + penalty
        return loss

    def compute_loss_derivative(self) -> np.ndarray:
        """
        Computes the derivative of the loss with respect to n for each domain.

        Returns:
            A numpy array of derivatives dL_k/dn for each domain.
        """
        assert self.scaling_law_params is not None

        if self.use_same_step_size:
            # In this case self.counts cannot be used. Only per_step_counts contains the
            # "adjusted" distribution of tokens.
            counts_over_time = np.array(self.per_step_counts)
            n_k = np.sum(counts_over_time, axis=0)
        else:
            n_k = self.counts.copy()

        if self.count_normalizer is not None and self.count_normalizer > 1:
            n_k = n_k / self.count_normalizer

        # Extract parameters
        log_beta_k, log_epsilon_k, alpha_k = self.scaling_law_params.T
        beta_k = np.exp(log_beta_k)
        epsilon_k = np.exp(log_epsilon_k)

        # Handle domains with n_k > 0 to avoid division by zero
        mask = n_k > 0

        # Compute L_k(n) for n_k > 0
        L_k_n = np.zeros_like(n_k, dtype=np.float64)
        L_k_n[mask] = epsilon_k[mask] + beta_k[mask] * n_k[mask] ** (-alpha_k[mask])

        # Compute derivative dL_k/dn for n_k > 0
        dL_dn = np.zeros_like(n_k, dtype=np.float64)
        dL_dn[mask] = -(1 / n_k[mask]) * alpha_k[mask] * (L_k_n[mask] - epsilon_k[mask])

        return dL_dn

    def compute_rho_t(self, dL_dn: np.ndarray) -> None:
        """
        Computes the preference distribution rho_k(t) based on the derivative of the loss.

        Args:
            dL_dn: A numpy array of derivatives of the loss with respect to n for each domain.
        """
        assert self.h_t is not None and self.mu_k is not None
        # Compute lambda_k(t)
        lambda_k_t = self.h_t.copy() ** self.s

        rho_num = self.mu_k * lambda_k_t * (-dL_dn)
        rho_num = np.maximum(rho_num, 0)

        rho_den: float | np.floating = np.sum(rho_num)
        if rho_den > 0:
            self.rho_t = rho_num / rho_den  # TODO(MaxiBoether): test without this normalizaiton.
            # logger.debug(f"Computed rho_t = {self.rho_t}")
        else:
            # Handle the case where denominator is zero
            self.rho_t = self.mu_k.copy() / len(self.counts)
            # logger.debug(f"Computed special rho_t = {self.rho_t}")

    def update_pi_t(self) -> None:
        """
        Updates the data policy pi_k(t) based on rho_k(t) and the moving average pī_k(t-1).
        """
        assert self.mu_k is not None

        self.pi_bar_t_minus_1 = self.pi_bar_t_minus_1 if self.pi_bar_t_minus_1 is not None else self.mu_k.copy()

        pi_t = self.gamma2 * self.rho_t + (1 - self.gamma2) * self.pi_bar_t_minus_1
        pi_t /= np.sum(pi_t)

        self.pi_t = pi_t

    def enforce_min_probability(self) -> None:
        """
        Enforces the minimum probability delta_min for domains that have been sampled at least once.

        Args:
            pi_t: The data policy pi_k(t) before enforcing minimum probabilities.

        Returns:
            The data policy pi_k(t) after enforcing minimum probabilities.
        """
        # Note that this currently does NOT align with the papers implementation.
        # The paper IMPLEMENTATION uses a different redistribution mechanism and clips rho_t,
        # while we currently clip pi_t.
        assert self.pi_t is not None and self.mu_k is not None

        pi_t_adjusted = self.pi_t.copy()

        # Create a mask for domains that have been sampled at least once
        sampled_mask = self.counts > 0

        # Enforce delta_min only on sampled domains
        pi_t_adjusted[sampled_mask] = np.maximum(pi_t_adjusted[sampled_mask], self.delta_min)

        # Re-normalize the probabilities
        total: float | np.floating = np.sum(pi_t_adjusted)
        if total > 0:
            pi_t_adjusted /= total
        else:
            # Handle zero total probability
            pi_t_adjusted = self.mu_k.copy() / len(self.counts)

        self.pi_t = pi_t_adjusted

    def update_pi_bar_t(self) -> None:
        """
        Updates the moving average pī_k(t) of the data policy.
        """
        assert self.pi_t is not None and self.pi_bar_t_minus_1 is not None

        weight = 1.0 / (self.total_steps + 1.0)
        self.pi_bar_t_minus_1 = weight * self.rho_t + (1 - weight) * self.pi_bar_t_minus_1
        assert self.pi_bar_t_minus_1 is not None  # mypy is really weird sometimes.
        self.pi_bar_t_minus_1 /= np.sum(self.pi_bar_t_minus_1)  # TODO(MaxiBoether): test without this normalizaiton.

    def _update_state(self, losses: np.ndarray, counts: np.ndarray) -> None:
        """
        Accumulates the losses and counts, adjusting internal arrays as needed to accommodate new domains.

        Args:
            losses: A numpy array of losses per domain.
            counts: A numpy array of counts per domain.
        """
        # We first need to catch the number of internal domains, otherwise we will not trigger the following condition.
        num_incoming_domains = len(losses)
        num_internal_domains = len(self.losses)

        if self.log_counts is not None:
            self.log_counts.append(counts.tolist())

        super()._update_state(losses, counts)

        if num_internal_domains < num_incoming_domains:
            # Expand per-domain data structures
            size_diff = num_incoming_domains - num_internal_domains
            logger.debug(f"Resizing structures in ADO algorithm. size_diff = {size_diff}")

            if self.h_t is not None:
                self.h_t = np.concatenate([self.h_t, np.zeros(size_diff, dtype=self.h_t.dtype)])
            if self.mu_k is not None:
                self.mu_k = np.concatenate([self.mu_k, np.zeros(size_diff, dtype=self.mu_k.dtype)])
            if self.pi_t is not None:
                self.pi_t = np.concatenate([self.pi_t, np.zeros(size_diff, dtype=self.pi_t.dtype)])
            if self.pi_bar_t_minus_1 is not None:
                self.pi_bar_t_minus_1 = np.concatenate(
                    [self.pi_bar_t_minus_1, np.zeros(size_diff, dtype=self.pi_bar_t_minus_1.dtype)]
                )

        # Keep track of per-step counts and losses
        # Adjust the lengths if necessary
        for idx, per_step_count in enumerate(self.per_step_counts):
            if len(per_step_count) < num_incoming_domains:
                size_diff = num_incoming_domains - len(per_step_count)
                self.per_step_counts[idx] = np.concatenate(
                    [self.per_step_counts[idx], np.zeros(size_diff, dtype=self.per_step_counts[idx].dtype)]
                )
                self.per_step_losses[idx] = np.concatenate(
                    [self.per_step_losses[idx], np.zeros(size_diff, dtype=self.per_step_losses[idx].dtype)]
                )

        # Compute NORMALIZED per-step losses per domain (avoiding division by zero)
        per_step_losses = np.divide(losses, counts, out=np.zeros_like(losses, dtype=losses.dtype), where=counts != 0)
        self.per_step_losses.append(per_step_losses)

        if self.use_same_step_size:
            num_valid_domains: int = np.sum(self.counts > 0)
            assert num_valid_domains > 0
            total_tokens = counts.sum()
            increment = np.zeros_like(counts, dtype=counts.dtype)
            increment[self.counts > 0] = int(total_tokens)
            self.per_step_counts.append(increment)
        else:
            self.per_step_counts.append(counts.copy())
