from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from mixtera.core.algo.dynamic_mixing.dynamic_mixing import DynamicMixingAlgorithm
from mixtera.core.query.mixture.mixture import Mixture
from mixtera.core.query.mixture.mixture_key import MixtureKey
from mixtera.core.query.mixture.static_mixture import StaticMixture

if TYPE_CHECKING:
    from mixtera.core.datacollection.index import ChunkerIndex


class DynamicMixture(Mixture):
    """Mixture class that uses a dynamic mixing algorithm."""

    def __init__(
        self, chunk_size: int, initial_mixture: Mixture, mixing_alg: DynamicMixingAlgorithm, strict: bool = True
    ) -> None:
        """
        Initializer for DynamicMixture.

        Args:
            chunk_size: the size of a chunk in number of instances
            initial_mixture: Another mixture object that defines which mixture to start with.
                Most likely, you want a custom static mixture or a inferring mixture here.
            mixing_alg: An instance of a DynamicMixingAlgorithm that defines how to calculate the dynamic mixture.
        """
        super().__init__(chunk_size, strict=strict)
        if initial_mixture.chunk_size != chunk_size:
            logger.warning(
                f"DynamicMixture chunk size is {chunk_size}, "
                + f"initial_mixture chunk size is {initial_mixture.chunk_size}. Adjusting."
            )
            initial_mixture.chunk_size = chunk_size

        self._current_mixture = initial_mixture
        self._mixing_alg = mixing_alg
        self._informed_alg = False  # Indicates whether we have reported the initial distribution to the algorithm.
        self._key_id_map: dict[MixtureKey, int] | None = None
        self._id_key_map: dict[int, MixtureKey] | None = None

    def __str__(self) -> str:
        """String representation of this mixture object."""
        return str(
            {
                "mixture": "dynamic_mixture",
                "chunk_size": self.chunk_size,
                "algo": str(self._mixing_alg),
                "current_mixture": str(self._current_mixture),
                "strict": {self.strict},
            }
        )

    def mixture_in_rows(self) -> dict[MixtureKey, int]:
        return self._current_mixture.mixture_in_rows()

    def process_index(self, chunker_index: "ChunkerIndex") -> None:
        # Mostly useful for using an InferringMixture as the initial mixture.
        self._current_mixture.process_index(chunker_index)

    def process_id_map(self, key_id_map: dict[MixtureKey, int]) -> None:
        self._key_id_map = key_id_map
        self._id_key_map = {value: key for key, value in key_id_map.items()}

        logger.debug("Updated ID map of dynamic mixture.")

        if not self._informed_alg:
            # First `inform` is called (which will potentially build the initial mixture), then this function is called.
            # If after that we still have a None mixture, the dynamic mixture does not work.
            mixture = self.mixture_in_rows()
            assert mixture is not None, "mixture_in_rows is None even after informing about the id map."
            initial_mix = np.zeros(max(key_id_map.values()) + 1, dtype=self._mixing_alg.initial_mixture.dtype)
            for key, value in mixture.items():
                initial_mix[self._key_id_map[key]] = value

            initial_mix = initial_mix / np.sum(initial_mix)
            assert np.isclose(initial_mix.sum(), 1), f"Initial mixture sums to {initial_mix.sum()} instead of 1."

            self._mixing_alg.initial_mixture = initial_mix

            self._informed_alg = True

            logger.debug(f"Informed the algorithm about the initial mixture of length = {len(initial_mix)}")

    def _process_losses(self, losses: np.ndarray, counts: np.ndarray, mixture_id: int) -> None:
        assert self._key_id_map is not None and self._id_key_map is not None
        assert self._informed_alg
        assert mixture_id >= 0

        if (mixture_np := self._mixing_alg.process_losses(losses, counts, mixture_id)) is not None:
            logger.debug(
                "Updated dynamic mixing algorithm.\n"
                + f"key_id_map = {self._key_id_map}\n id_key_map = {self._id_key_map}\n mixture_np = {mixture_np}"
            )
            assert np.isclose(mixture_np.sum(), 1), (
                f"Mixture result is {mixture_np}, which sums to {mixture_np.sum()} instead of 1. "
                + "There is an issue in the dynamic mixing algorithm."
            )
            weight_map = {self._id_key_map[idx]: val for idx, val in enumerate(mixture_np) if val > 0}
            logger.debug(f"weight_map = {weight_map}")
            self._current_mixture = StaticMixture(self.chunk_size, weight_map)
            logger.debug(f"New mixture is {self._current_mixture}")
