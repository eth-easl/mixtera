from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from loguru import logger
from mixtera.core.datacollection.index import infer_mixture_from_chunkerindex

if TYPE_CHECKING:
    from mixtera.core.datacollection.index import ChunkerIndex

# class MixtureKey(ABC):
#
#     def __eq__(self, other):
#         raise NotImplementedError("Method must be implemented in subclass!")
#
#     def __hash__(self):
#         raise NotImplementedError("Method must be implemented in subclass!")

# class StringMixtureKey(MixtureKey):
#     """
#     The StringMixtureKey represents a chunk index key which receives one and only one value per property. Equality
#     between two keys implies that they have the same exact keys and value for each key. This type of key also accepts
#     a single property with the name "ANY"
#     """
#
#     def __init__(self, property_names: list[str], property_values: list[list[str | int | float]]) -> None:
#         """
#         Creates a StringMixtureKey object.
#
#         Args:
#             property_names: a list with the property names
#             property_values: a list of lists with the property values
#
#         Returns:
#             A string that can be used in a ChunkerIndex to identify ranges fulfilling a certain property
#         """


class Mixture(ABC):
    """Base Mixture class."""

    def __init__(self, chunk_size: int) -> None:
        """
        Base initialize for a Mixture object.

        Args:
            chunk_size: the size of a chunk in number of instances
        """
        self.chunk_size = chunk_size

    def __str__(self) -> str:
        """String representation of this mixture object."""
        return f'{{"mixture": "base_mixture", "chunk_size": {self.chunk_size}}}'

    @abstractmethod
    def mixture_in_rows(self) -> dict[str, int]:
        """
        Returns the mixture dictionary:
        {
            "component_0" : number_of_instances_for_component_0,
            ...
        }

        where:
            'component_0' is a serialized representation of some mixture component, e.g.
                "property0:value0;property1:value1;...", and
            'number_of_instances_for_component_0' is the concrete number of instances per chunk for this particular
                mixture component, e.g. 200.

        Returns:
            The mixture dictionary.
        """
        raise NotImplementedError("Method must be implemented in subclass!")

    @abstractmethod
    def inform(self, chunker_index: "ChunkerIndex") -> None:
        """
        Function that is called to inform the mixture class about the overall chunker index, i.e.,
        the overall distribution in the QueryResult.
        """
        raise NotImplementedError("Method must be implemented in subclass!")


class ArbitraryMixture(Mixture):
    """
    This is a mixture that allows for chunks to be created without any particular mixture.
    This mixture makes no guarantees at all and yields chunks that may contain spurious correlations,
    e.g., only data from one type. If you want a more balanced chunk without specifying a mixture,
    consider using the `InferringMixture`.
    """

    def mixture_in_rows(self) -> dict[str, int]:
        return {}

    def __str__(self) -> str:
        """String representation of this mixture object."""
        return f'{{"mixture": "arbitrary_mixture", "chunk_size": {self.chunk_size}}}'

    def inform(self, chunker_index: "ChunkerIndex") -> None:
        del chunker_index


class InferringMixture(Mixture):
    """
    This is a mixture that allows for chunks to be created without specifying a Mixture.
    Each chunk is represented with the same mixture that is in the overall QueryResult,
    to have a balanced sample per chunk.
    """

    def __init__(self, chunk_size: int) -> None:
        super().__init__(chunk_size)
        self._mixture: dict[str, int] = {}

    def mixture_in_rows(self) -> dict[str, int]:
        return self._mixture

    def __str__(self) -> str:
        """String representation of this mixture object."""
        return f'{{"mixture": "{self._mixture}", "chunk_size": {self.chunk_size}}}'

    def inform(self, chunker_index: "ChunkerIndex") -> None:
        logger.info("InferringMixture starts inferring mixture.")
        total, inferred_mixture_dict = infer_mixture_from_chunkerindex(chunker_index)
        logger.debug(f"total={total}, inferred_dict = {inferred_mixture_dict}")
        self._mixture = StaticMixture.parse_user_mixture(self.chunk_size, inferred_mixture_dict)
        logger.info("Mixture inferred.")


class StaticMixture(Mixture):
    """Mixture class that simply stores a predefined mixture."""

    def __init__(self, chunk_size: int, mixture: dict[str, float]) -> None:
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
    def parse_user_mixture(chunk_size: int, user_mixture: dict[str, float]) -> dict[str, int]:
        """Given a chunk size and user mixture, return an internal adjusted representation
        that handles rounding errors and that adheres to the chunk size."""
        mixture = {key: int(chunk_size * val) for key, val in user_mixture.items()}

        # Ensure approximation errors do not affect final chunk size
        if (diff := chunk_size - sum(mixture.values())) > 0:
            mixture[list(mixture.keys())[0]] += diff

        return mixture

    def __str__(self) -> str:
        """String representation of this mixture object."""
        return f'{{"mixture": {self._mixture}, "chunk_size": {self.chunk_size}}}'

    def mixture_in_rows(self) -> dict[str, int]:
        return self._mixture

    def inform(self, chunker_index: "ChunkerIndex") -> None:
        del chunker_index
