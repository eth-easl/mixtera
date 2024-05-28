from abc import ABC, abstractmethod


class Mixture(ABC):
    """Base Mixture class."""

    def __init__(self, chunk_size: int) -> None:
        """
        Base initialize for a Mixture object.

        Args:
            chunk_size: the size of a chunk in number of instances
        """
        self.chunk_size = chunk_size

    def __str__(self):
        """String representation of this mixture object."""
        raise NotImplementedError("Method must be implemented in subclass!")

    @abstractmethod
    def get_mixture(self) -> dict[str, int]:
        """
        Returns the mixture dictionary:
        {
            "serialized_condition_0": number_of_instances_for_partition_0,
            ...
        }

        Returns:
            The mixture dictionary.

        """
        raise NotImplementedError("Method must be implemented in subclass!")


class StaticMixture(Mixture):
    """Mixture class that simply stores a predefined mixture."""

    def __init__(self, chunk_size: int, mixture: dict[str, float]) -> None:
        super().__init__(chunk_size)
        self._mixture = {key: int(chunk_size * val) for key, val in mixture.items()}

        # Ensure approximation errors do not affect final chunk size
        diff = chunk_size - sum(self._mixture.values())
        if diff > 0:
            self._mixture[list(self._mixture.keys())[0]] += diff

    def __str__(self):
        """String representation of this mixture object."""
        return f'{{"mixture": {self._mixture}, "chunk_size": {self.chunk_size}}}'

    def get_mixture(self) -> dict[str, int]:
        return self._mixture
