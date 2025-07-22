from typing import TYPE_CHECKING

from loguru import logger

from mixtera.core.datacollection.index import infer_mixture_from_chunkerindex
from mixtera.core.query.mixture.mixture import Mixture
from mixtera.core.query.mixture.mixture_key import MixtureKey
from mixtera.core.query.mixture.static_mixture import StaticMixture

if TYPE_CHECKING:
    from mixtera.core.datacollection.index import ChunkerIndex


class InferringMixture(Mixture):
    """
    This is a mixture that allows for chunks to be created without specifying a Mixture.
    Each chunk is represented with the same mixture that is in the overall QueryResult,
    to have a balanced sample per chunk.
    """

    def __init__(self, chunk_size: int, strict: bool = True) -> None:
        super().__init__(chunk_size, strict=strict)
        self._mixture: dict[MixtureKey, int] = {}

    def mixture_in_rows(self) -> dict[MixtureKey, int]:
        return self._mixture

    def __str__(self) -> str:
        """String representation of this mixture object."""
        return f'{{"mixture": "{self._mixture}", "chunk_size": {self.chunk_size}, "strict": {self.strict}}}'

    def process_index(self, chunker_index: "ChunkerIndex") -> None:
        logger.info("InferringMixture starts inferring mixture.")
        total, inferred_mixture_dict = infer_mixture_from_chunkerindex(chunker_index)
        logger.debug(f"total={total}, inferred_dict = {inferred_mixture_dict}")

        if total == 0:
            assert (
                not inferred_mixture_dict
            ), f"Inconsistent state: total = 0, inferred_mixture_dict = {inferred_mixture_dict}"
            logger.warning("Cannot infer mixture since chunker index is empty.")
            self._mixture = {}
            return

        assert (
            total > 0 and len(inferred_mixture_dict.keys()) > 0
        ), f"Inconsistent state: total = {total}, inferred_mixture_dict={inferred_mixture_dict}"

        self._mixture = StaticMixture.parse_user_mixture(self.chunk_size, inferred_mixture_dict)
        logger.info("Mixture inferred.")
