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
