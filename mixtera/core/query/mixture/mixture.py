from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from mixtera.core.query.mixture.mixture_key import MixtureKey
from mixtera.network.client.client_feedback import ClientFeedback

if TYPE_CHECKING:
    from mixtera.core.datacollection.index import ChunkerIndex


class Mixture(ABC):
    """Base Mixture class."""

    def __init__(self, chunk_size: int, strict: bool = True) -> None:
        """
        Base initialize for a Mixture object.

        Args:
            chunk_size: the size of a chunk in number of instances
            strict: best_effort mode if False, strict mode if True
        """
        self.chunk_size = chunk_size
        self.current_step = 0
        self.strict = strict

    def __str__(self) -> str:
        """String representation of this mixture object."""
        return f'{{"mixture": "base_mixture", "chunk_size": {self.chunk_size}, "strict": {self.strict}}}'

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
    def process_index(self, chunker_index: "ChunkerIndex") -> None:
        """
        Function that is called to inform the mixture class about the overall chunker index, i.e.,
        the overall distribution in the QueryResult.
        """
        raise NotImplementedError("Method must be implemented in subclass!")

    def process_client_feedback(self, feedback: ClientFeedback) -> None:
        """
        Updates the mixture according to the received feedback.

        Args:
            feedback: The received feedback from trainig.
        """
        self._update_training_step(feedback.training_steps)
        if feedback.counts is not None and feedback.losses is not None:
            self._process_losses(feedback.losses, feedback.counts, feedback.mixture_id)

    def _update_training_step(self, training_steps: int) -> None:
        """
        Updates the current training step according to the received feedback.
        The training steps can only increase.

        Args:
            training_steps: The current training step of the model.
        """
        self.current_step = max(self.current_step, training_steps)

    def stringified_mixture(self) -> dict[str, int]:
        """
        Helper fuction that returns the current mixture representation using string keys.
        """
        return {str(key): val for key, val in self.mixture_in_rows().items()}

    def process_id_map(self, key_id_map: dict[MixtureKey, int]) -> None:
        del key_id_map

    def _process_losses(self, losses: np.ndarray, counts: np.ndarray, mixture_id: int) -> None:
        del losses
        del counts
        del mixture_id

    def write_logs(self) -> None:
        pass
