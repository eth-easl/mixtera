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
            mixture: a dictionary that points from mixture components to concentration/mass in mixture of the form:
                {
                   "property0:value0;property1:value1;..." : 0.2,
                   "property0:value1;property1:value1" : 0.1,
                   "property0:value2": 0.35
                   ...
                }
        """
        super().__init__(chunk_size)
        self._mixture = StaticMixture.parse_user_mixture(chunk_size, mixture)

    @staticmethod
    def parse_user_mixture(chunk_size: int, user_mixture: dict[MixtureKey, float]) -> dict[MixtureKey, int]:
        """Given a chunk size and user mixture, return an internal adjusted representation
        that handles rounding errors and that adheres to the chunk size."""
        for key, val in user_mixture.items():
            assert val >= 0, "Mixture values must be non-negative."
            assert isinstance(key, MixtureKey), "Mixture keys must be of type MixtureKey."


        # todo think about this.
        
        # Calculate ideal counts and their floor counts
        ideal_counts = {key: chunk_size * val for key, val in user_mixture.items()}
        floor_counts = {key: int(ideal_count) for key, ideal_count in ideal_counts.items()}
        fractions = {key: ideal_counts[key] - floor_counts[key] for key in user_mixture.keys()}

        # Calculate total floor count and the difference to distribute
        total_floor_count = sum(floor_counts.values())
        diff = chunk_size - total_floor_count  # Number of counts to adjust

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
        elif diff < 0:
            # Reduce counts from items with the smallest fractional parts
            sorted_keys = sorted(fractions.keys(), key=lambda k: fractions[k])
            index = 0
            num_keys = len(sorted_keys)
            while diff < 0:
                key = sorted_keys[index % num_keys]
                if floor_counts[key] > 0:
                    floor_counts[key] -= 1
                    diff += 1
                index += 1

        # The floor_counts now sum up to chunk_size
        mixture = floor_counts
        return mixture


    def __str__(self) -> str:
        """String representation of this mixture object."""
        return f'{{"mixture": {self._mixture}, "chunk_size": {self.chunk_size}}}'

    def mixture_in_rows(self) -> dict[MixtureKey, int]:
        return self._mixture

    def process_index(self, chunker_index: "ChunkerIndex") -> None:
        del chunker_index
