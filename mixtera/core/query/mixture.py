from abc import ABC, abstractmethod

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


class ArbitraryMixture(Mixture):
    """This is a placeholder mixture that allows for chunks to be created without any particular mixture."""

    def mixture_in_rows(self) -> dict[str, int]:
        return {}

    def __str__(self) -> str:
        """String representation of this mixture object."""
        return f'{{"mixture": "arbitrary_mixture", "chunk_size": {self.chunk_size}}}'


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
        self._mixture = {key: int(chunk_size * val) for key, val in mixture.items()}

        # Ensure approximation errors do not affect final chunk size
        if (diff := chunk_size - sum(self._mixture.values())) > 0:
            self._mixture[list(self._mixture.keys())[0]] += diff

    def __str__(self) -> str:
        """String representation of this mixture object."""
        return f'{{"mixture": {self._mixture}, "chunk_size": {self.chunk_size}}}'

    def mixture_in_rows(self) -> dict[str, int]:
        return self._mixture
