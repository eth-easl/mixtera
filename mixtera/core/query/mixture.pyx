# mixtera/core/query/mixture.pyx

from cpython cimport Py_hash_t  # Import Py_hash_t for hash values

import cython


@cython.cfunc
@cython.locals(d=dict)
def hash_dict(d):
    """
    Generate a hash from a dictionary.
    """
    cdef object items = tuple(sorted(d.items()))
    cdef Py_hash_t hash_value = hash(items)
    # Ensure that the hash value is not -1 or the sentinel value (-2)
    if hash_value == -1 or hash_value == -2:
        hash_value = -3  # Adjust to another value
    return hash_value

cdef class MixtureKey:
    cdef dict properties
    cdef Py_hash_t _hash

    def __init__(self, dict properties):
        self.properties = properties
        self._hash = -2  # Use -2 as the sentinel value for uninitialized hash

    def __hash__(self):
        if self._hash == -2:  # Check for sentinel value
            self._hash = hash_dict(self.properties)
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, MixtureKey):
            return False

        cdef str k
        cdef object v, other_v

        for k, v in self.properties.items():
            if k not in other.properties:
                return False
            other_v = other.properties[k]
            if not set(v).intersection(other_v) and (len(v) > 0 or len(other_v) > 0):
                return False
        return True

    def __lt__(self, other):
        if not isinstance(other, MixtureKey):
            return NotImplemented

        cdef bint is_less_than, is_greater_than
        is_less_than, is_greater_than = self._compare_properties(other)

        if is_less_than and not is_greater_than:
            return True
        if is_greater_than and not is_less_than:
            return False

        return len(self.properties) < len(other.properties)

    def __gt__(self, other):
        if not isinstance(other, MixtureKey):
            return NotImplemented

        cdef bint is_less_than, is_greater_than
        is_less_than, is_greater_than = self._compare_properties(other)

        if is_greater_than and not is_less_than:
            return True
        if is_less_than and not is_greater_than:
            return False

        return len(self.properties) > len(other.properties)

    cpdef tuple _compare_properties(self, MixtureKey other):
        cdef bint is_less_than = False
        cdef bint is_greater_than = False
        cdef str k
        cdef object v

        for k, v in self.properties.items():
            if k not in other.properties:
                continue
            if len(v) < len(other.properties[k]):
                is_less_than = True
            elif len(v) > len(other.properties[k]):
                is_greater_than = True

        return is_less_than, is_greater_than

    def __str__(self):
        return ";".join([
            f'{k}:{":".join([str(x) for x in v])}'
            for k, v in sorted(self.properties.items())
        ])

    def __repr__(self):
        return str(self)
