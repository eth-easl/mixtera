from collections import defaultdict
from typing import Any

from mixtera.core.datacollection.index import ChunkerIndex

# Note that these functions cannot be nested lambdas or nested functions since they cannot be pickled
# when using 'spawn' in multiprocessing.


def create_inner_dict() -> defaultdict[Any, list]:
    """
    Creates and returns a `defaultdict` with a default factory of `list`.
    This represents the innermost level of the index structure where each
    feature value points to a list of payloads (row indices or row ranges).

    Returns:
        defaultdict[Any, list]: A `defaultdict` mapping keys to lists.
    """
    return defaultdict(list)


def create_mid_dict() -> defaultdict[Any, defaultdict]:
    """
    Creates and returns a `defaultdict` with a default factory that produces
    `defaultdict`s of lists. This represents the layer in the index structure
    where each file ID maps to a `defaultdict` that holds the payloads for
    that file.

    Returns:
        defaultdict[Any, defaultdict]: A `defaultdict` mapping keys to `defaultdict`s of lists.
    """
    return defaultdict(create_inner_dict)


def create_outer_dict() -> defaultdict[Any, defaultdict]:
    """
    Creates and returns a `defaultdict` with a default factory that produces
    `defaultdict`s of `defaultdict`s of lists. This represents the layer in
    the index structure where each dataset ID maps to a `defaultdict` that
    holds the file IDs and their associated payloads.

    Returns:
        defaultdict[Any, defaultdict]: A `defaultdict` mapping keys to `defaultdict`s of `defaultdict`s of lists.
    """
    return defaultdict(create_mid_dict)


def create_chunker_index() -> ChunkerIndex:
    """
    Creates an ChunkerIndex type.

    Returns: an ChunkerIndex object
    """
    return create_outer_dict()
