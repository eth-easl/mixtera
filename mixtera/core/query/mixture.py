from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from loguru import logger
from mixtera.core.datacollection.index import infer_mixture_from_chunkerindex
from mixtera.utils import hash_dict

if TYPE_CHECKING:
    from mixtera.core.datacollection.index import ChunkerIndex


class MixtureKey:
    __slots__ = ["properties", "_hash"]

    def __init__(self, properties: dict[str, list[str | int | float]]) -> None:
        """
        Initialize a MixtureKey object.

        Args:
            properties: a dictionary of properties for the mixture key
        """
        self.properties = properties
        self._hash: int | None = None

    def add_property(self, property_name: str, property_value: list[str | int | float]) -> None:
        self.properties[property_name] = property_value
        self._hash = None

    def __eq__(self, other: object) -> bool:
        #  TODO(#112): This is currently not commutative, i.e., a == b does not imply b == a
        if not isinstance(other, MixtureKey):
            return False

        # Note: Because this does not actually implement equality, we CANNOT use the hash here
        # (if it exists) to check whether we are equal. This is not equality! (Caused quite some debugging...)

        #  We compare the properties of the two MixtureKey objects
        for k, v in self.properties.items():
            #  If a property is not present in the other MixtureKey, we return False
            if k not in other.properties:
                return False
            #  If the values of the two properties do not have any intersection, we return False
            other_v = set(other.properties[k])
            if not set(v).intersection(other_v) and (len(v) > 0 or len(other_v) > 0):
                return False
        return True

    #  Mixture keys with multiple values for the same property are "greater" than those with a single value
    #  Where we count the number of values for each property (compare per property)
    def __lt__(self, other: object) -> bool:
        if not isinstance(other, MixtureKey):
            return NotImplemented

        is_less_than, is_greater_than = self._compare_properties(other)

        if is_less_than and not is_greater_than:
            return True
        if is_greater_than and not is_less_than:
            return False

        # If equal in all shared keys, compare the total number of properties
        return len(self.properties) < len(other.properties)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, MixtureKey):
            return NotImplemented

        is_less_than, is_greater_than = self._compare_properties(other)

        if is_greater_than and not is_less_than:
            return True
        if is_less_than and not is_greater_than:
            return False

        # If equal in all shared keys, compare the total number of properties
        return len(self.properties) > len(other.properties)

    def _compare_properties(self, other: "MixtureKey") -> tuple[bool, bool]:
        is_less_than = False
        is_greater_than = False

        for k, v in self.properties.items():
            if k not in other.properties:
                continue
            if len(v) < len(other.properties[k]):
                is_less_than = True
            elif len(v) > len(other.properties[k]):
                is_greater_than = True

        return is_less_than, is_greater_than

    def __hash__(self) -> int:
        #  Since we are want to use this class as a key in a dictionary, we need to implement the __hash__ method
        self._hash = self._hash if self._hash is not None else hash_dict(self.properties)
        return self._hash

    def __str__(self) -> str:
        #  We sort the properties to ensure that the string representation is deterministic
        return ";".join([f'{k}:{":".join([str(x) for x in sorted(v)])}' for k, v in sorted(self.properties.items())])

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
        self._mixture: dict[MixtureKey, int] = {}

    def mixture_in_rows(self) -> dict[MixtureKey, int]:
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
    def parse_user_mixture(chunk_size: int, user_mixture: dict[MixtureKey, float]) -> dict[MixtureKey, int]:
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


@dataclass
class MixtureNode:
    property_name: str
    components: List["Component"]


@dataclass
class Component:
    value: List[str | int | float]
    weight: float
    submixture: None | MixtureNode = None


class HierarchicalStaticMixture(Mixture):
    """Mixture class that simply stores a predefined mixture.
    Different from StaticMixture it receives the mixture combinations in a hierarchical manner."""

    def __init__(self, chunk_size: int, mixture: MixtureNode) -> None:
        """
        Initializer for HierarchicalStaticMixture. The portions of the components should add up to 1.

        Args:
            chunk_size: the size of a chunk in number of instances
            mixture: HierarchicalMixture that enables submixture definitions.
            ex: HierarchicalMixture(property_name="topic", components=[Component(value="law", weight=0.5),
            Component(value="medicine", weight=0.5)])
        """
        super().__init__(chunk_size)
        self._mixture = self.parse_mixture_node(chunk_size, mixture)

    def __str__(self) -> str:
        """String representation of this mixture object."""
        return f'{{"mixture": {self._mixture}, "chunk_size": {self.chunk_size}}}'

    def parse_mixture_node(self, chunk_size: int, user_mixture: MixtureNode) -> dict[MixtureKey, int]:
        formatted_user_mixture = self.convert_to_mixture_key_format(user_mixture)
        for key, val in formatted_user_mixture.items():
            assert val >= 0, "Mixture values must be non-negative."
            assert isinstance(key, MixtureKey), "Mixture keys must be of type MixtureKey."

        mixture = {key: int(chunk_size * val) for key, val in formatted_user_mixture.items()}

        # Ensure approximation errors do not affect final chunk size
        if (diff := chunk_size - sum(mixture.values())) > 0:
            mixture[list(mixture.keys())[0]] += diff

        return mixture

    def convert_to_mixture_key_format(self, mixture: MixtureNode) -> dict[MixtureKey, float]:
        mixture_keys = {}
        for component in mixture.components:
            if component.submixture is not None:
                result = self.convert_to_mixture_key_format(component.submixture)
                for key, value in result.items():
                    key.add_property(mixture.property_name, component.value)
                    mixture_keys[key] = value * component.weight
            else:
                mixture_keys[MixtureKey({mixture.property_name: component.value})] = component.weight
        return mixture_keys

    def mixture_in_rows(self) -> dict[MixtureKey, int]:
        return self._mixture

    def inform(self, chunker_index: "ChunkerIndex") -> None:
        del chunker_index
