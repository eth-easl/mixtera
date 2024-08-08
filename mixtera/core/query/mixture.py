from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from loguru import logger
from mixtera.core.datacollection.index import infer_mixture_from_chunkerindex
from mixtera.utils import hash_dict

if TYPE_CHECKING:
    from mixtera.core.datacollection.index import ChunkerIndex


class MixtureKey:

    def __init__(self, properties: dict[str, list[str | int | float]]) -> None:
        """
        Initialize a MixtureKey object.

        Args:
            properties: a dictionary of properties for the mixture key
        """
        self.properties = properties

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MixtureKey):
            return False

        #  If there are no overlapping properties, we return False
        if not set(self.properties.keys()).intersection(other.properties.keys()):
            return False

        #  We compare the properties of the two MixtureKey objects
        for k, v in self.properties.items():
            #  If a property is not present in the other MixtureKey, we return False
            if k not in other.properties:
                return False
            #  If the values of the two properties do not have any intersection, we return False
            if not set(v).intersection(other.properties[k]):
                return False
        return True

    #  Mixture keys with multiple values for the same property are "greater" than those with a single value
    #  Where we count the number of values for each property (compare per property)
    def __lt__(self, other: object) -> bool:
        if not isinstance(other, MixtureKey):
            return False

        less_than_count = 0
        for k, v in self.properties.items():
            if k not in other.properties:
                continue
            if len(v) < len(other.properties[k]):
                less_than_count += 1

        if less_than_count > 0:
            return True

        return len(self.properties) < len(other.properties)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, MixtureKey):
            return False

        greater_than_count = 0
        for k, v in self.properties.items():
            if k not in other.properties:
                continue
            if len(v) > len(other.properties[k]):
                greater_than_count += 1

        if greater_than_count > 0:
            return True

        return len(self.properties) > len(other.properties)

    def __hash__(self) -> int:
        return hash_dict(self.properties)

    def __str__(self) -> str:
        #  We sort the properties to ensure that the string representation is deterministic
        return ";".join([f'{k}:{":".join([str(x) for x in v])}' for k, v in sorted(self.properties.items())])

    def __repr__(self) -> str:
        return str(self)


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
    def mixture_in_rows(self) -> dict[MixtureKey, int]:
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

    def mixture_in_rows(self) -> dict[MixtureKey, int]:
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
    def parse_user_mixture(chunk_size: int, user_mixture: dict[str, float]) -> dict[str, int]:
        """Given a chunk size and user mixture, return an internal adjusted representation
        that handles rounding errors and that adheres to the chunk size."""
        for key, val in user_mixture.items():
            assert val >= 0, "Mixture values must be non-negative."
            assert isinstance(key, MixtureKey), "Mixture keys must be of type MixtureKey."

        mixture = {key: int(chunk_size * val) for key, val in user_mixture.items()}

        # Ensure approximation errors do not affect final chunk size
        if (diff := chunk_size - sum(mixture.values())) > 0:
            mixture[list(mixture.keys())[0]] += diff

        return mixture

    def __str__(self) -> str:
        """String representation of this mixture object."""
        return f'{{"mixture": {self._mixture}, "chunk_size": {self.chunk_size}}}'

    def mixture_in_rows(self) -> dict[MixtureKey, int]:
        return self._mixture

    def inform(self, chunker_index: "ChunkerIndex") -> None:
        del chunker_index
