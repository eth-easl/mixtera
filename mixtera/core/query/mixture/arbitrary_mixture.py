from typing import TYPE_CHECKING

from mixtera.core.query.mixture import Mixture, MixtureKey

if TYPE_CHECKING:
    from mixtera.core.datacollection.index import ChunkerIndex


class ArbitraryMixture(Mixture):
    """
    This is a mixture that allows for chunks to be created without any particular mixture.
    This mixture makes no guarantees at all and yields chunks that may contain spurious correlations,
    e.g., only data from one type. If you want a more balanced chunk without specifying a mixture,
    consider using the `InferringMixture`.
    """

    def mixture_in_rows(self) -> dict[MixtureKey, int]:
        return {}

    def __str__(self) -> str:
        """String representation of this mixture object."""
        return f'{{"mixture": "arbitrary_mixture", "chunk_size": {self.chunk_size}}}'

    def inform(self, chunker_index: "ChunkerIndex") -> None:
        del chunker_index
