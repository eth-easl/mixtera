"""
This submodule contains general utility functions
"""

from .prefetch_iterator import PrefetchFirstItemIterator
from .tokenizing_iterator import ThreadedTokenizingIterator, TokenizingIterator
from .utils import (  # noqa: F401
    DummyPool,
    defaultdict_to_dict,
    deserialize_chunker_index,
    distribute_by_ratio,
    flatten,
    hash_dict,
    is_on_github_actions,
    merge_sorted_lists,
    numpy_to_native,
    numpy_to_native_type,
    run_async_until_complete,
    seed_everything_from_list,
    serialize_chunker_index,
    wait_for_key_in_dict,
)

__all__ = [
    "defaultdict_to_dict",
    "distribute_by_ratio",
    "flatten",
    "numpy_to_native_type",
    "run_async_until_complete",
    "wait_for_key_in_dict",
    "hash_dict",
    "seed_everything_from_list",
    "is_on_github_actions",
    "merge_sorted_lists",
    "numpy_to_native",
    "DummyPool",
    "serialize_chunker_index",
    "deserialize_chunker_index",
    "PrefetchFirstItemIterator",
    "TokenizingIterator",
    "ThreadedTokenizingIterator",
]
