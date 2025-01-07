from math import isclose
from typing import TYPE_CHECKING

from mixtera.core.query.mixture.mixture import Mixture
from mixtera.core.query.mixture.mixture_key import MixtureKey

if TYPE_CHECKING:
    from mixtera.core.datacollection.index import ChunkerIndex


class StaticMixture(Mixture):
    """Mixture class that simply stores a predefined mixture."""

    def __init__(self, chunk_size: int, mixture: dict[MixtureKey, float]) -> None:
        """
        Initializer for StaticMixture.

        Args:
            chunk_size: the size of a chunk in number of instances
            mixture: a dictionary that points from mixture components to "probability" in mixture of the form:
                {
                   "property0:value0;property1:value1;..." : 0.2,
                   "property0:value1;property1:value1" : 0.1,
                   "property0:value2": 0.35
                   ...
                }
                    Needs to sum to 1.
        """
        super().__init__(chunk_size)

        total_weight = sum(mixture.values())
        if not isclose(total_weight, 1.0, rel_tol=1e-7):
            raise ValueError(f"Your mixture sums up to {total_weight} != 1.0")

        # Renormalize to reduce deviance from 1.0 further
        mixture = {key: val / total_weight for key, val in mixture.items()}

        self._mixture = StaticMixture.parse_user_mixture(chunk_size, mixture)

    @staticmethod
    def parse_user_mixture(chunk_size: int, user_mixture: dict[MixtureKey, float]) -> dict[MixtureKey, int]:
        """Given a chunk size and user mixture, return an internal adjusted representation
        that handles rounding errors and that adheres to the chunk size."""
        for key, val in user_mixture.items():
            assert val >= 0, "Mixture values must be non-negative."
            assert isinstance(key, MixtureKey), "Mixture keys must be of type MixtureKey."

        ideal_counts = {key: chunk_size * val for key, val in user_mixture.items()}
        floor_counts = {key: int(ideal_count) for key, ideal_count in ideal_counts.items()}
        fractions = {key: ideal_counts[key] - floor_counts[key] for key in user_mixture.keys()}

        # Calculate total floor count and the difference to distribute
        total_floor_count = sum(floor_counts.values())
        diff = chunk_size - total_floor_count  # Number of counts to adjust

        assert diff >= 0, f"Unexpected diff = {diff}. Did the weights sum up to 1?"
        # largest remainders method: https://en.wikipedia.org/wiki/Quota_method
        if diff > 0:
            # Distribute additional counts to items with the largest fractional parts
            sorted_keys = sorted(fractions.keys(), key=lambda k: fractions[k], reverse=True)
            index = 0
            num_keys = len(sorted_keys)
            while diff > 0:
                key = sorted_keys[index % num_keys]
                floor_counts[key] += 1
                diff -= 1
                index += 1

        assert sum(floor_counts.values()) == chunk_size, f"floor_counts sum up to {sum(floor_counts.values())}"
        return floor_counts

    def __str__(self) -> str:
        """String representation of this mixture object."""
        return f'{{"mixture": {self._mixture}, "chunk_size": {self.chunk_size}}}'

    def mixture_in_rows(self) -> dict[MixtureKey, int]:
        return self._mixture

    def process_index(self, chunker_index: "ChunkerIndex") -> None:
        del chunker_index
