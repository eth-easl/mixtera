from mixtera.utils import hash_dict


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

    def add_property(self, property_name: str, property_values: list[str | int | float]) -> None:
        self.properties[property_name] = property_values
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
    def __lt__(self, other: "MixtureKey") -> bool:
        if not isinstance(other, MixtureKey):
            return NotImplemented

        # Compare number of properties
        len_self_properties = len(self.properties)
        len_other_properties = len(other.properties)
        if len_self_properties != len_other_properties:
            return len_self_properties < len_other_properties

        # Number of properties is equal
        # Get sorted list of property keys
        self_keys = sorted(self.properties)
        other_keys = sorted(other.properties)

        # Compare property keys
        for self_key, other_key in zip(self_keys, other_keys):
            if self_key != other_key:
                return self_key < other_key

            # Property names are the same, compare number of values
            len_self_values = len(self.properties[self_key])
            len_other_values = len(other.properties[other_key])
            if len_self_values != len_other_values:
                return len_self_values < len_other_values

        # Compare property values lexicographically
        for key in self_keys:
            self_values = sorted(map(str, self.properties[key]))
            other_values = sorted(map(str, other.properties[key]))
            if self_values != other_values:
                return self_values < other_values

        # All properties and values are equal, compare string representations
        return str(self) < str(other)

    def __hash__(self) -> int:
        #  Since we are want to use this class as a key in a dictionary, we need to implement the __hash__ method
        self._hash = self._hash if self._hash is not None else hash_dict(self.properties)
        return self._hash

    def __str__(self) -> str:
        #  We sort the properties to ensure that the string representation is deterministic
        return ";".join([f'{k}:{":".join([str(x) for x in sorted(v)])}' for k, v in sorted(self.properties.items())])

    def __repr__(self) -> str:
        return str(self)
