from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from mixtera.core.query.mixture.mixture_key import MixtureKey

if TYPE_CHECKING:
    from mixtera.core.datacollection.index import ChunkerIndex


class Mixture(ABC):
    """Base Mixture class."""

    def __init__(self, chunk_size: int) -> None:
        """
        Base initialize for a Mixture object.

        Args:
            chunk_size: the size of a chunk in number of instances
        """
        self.chunk_size = chunk_size
        self.current_step = 0

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

    def inform_training_step(self, training_steps: int) -> bool:
        """
        Updates the current mixture according to the received training step information.

        Args:
            training_steps: The current training step of the model.
        """
        assert self.current_step <= training_steps, "The mixture schedule is beyond the received feedback."

        self.current_step = training_steps
        return True

    def stringified_mixture(self) -> dict[str, int]:
        """
        Helper fuction that returns the current mixture representation using string keys.
        """
        return {str(key): val for key, val in self.mixture_in_rows().items()}
