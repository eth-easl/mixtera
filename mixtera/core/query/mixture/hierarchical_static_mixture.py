from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from mixtera.core.query.mixture import Mixture, MixtureKey

if TYPE_CHECKING:
    from mixtera.core.datacollection.index import ChunkerIndex


@dataclass
class MixtureNode:
    property_name: str
    components: list["Component"]


@dataclass
class Component:
    values: List[str | int | float]
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
            Component(values=["medicine"], weight=0.5)])
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
                    key.add_property(mixture.property_name, component.values)
                    mixture_keys[key] = value * component.weight
            else:
                mixture_keys[MixtureKey({mixture.property_name: component.values})] = component.weight
        return mixture_keys

    def mixture_in_rows(self) -> dict[MixtureKey, int]:
        return self._mixture

    def inform(self, chunker_index: "ChunkerIndex") -> None:
        del chunker_index
