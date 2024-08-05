import asyncio
import hashlib
import os
import random
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, List, Tuple, Union

import numpy as np
import portion as P


def flatten(non_flat_list: List[List[Any]]) -> List[Any]:
    return [item for sublist in non_flat_list for item in sublist]


def ranges(nums: List[int]) -> List[Tuple[int, int]]:
    """
    This method compresses a list of integers into a list of ranges (lower bound
    inclusve, upper bound exclusive). E.g. [1,2,3,5,6] --> [(1,4), (5,7)].

    Args:
        nums: The original list of ranges to compress. This is a list of ints.

    Returns:
        A list of compressed ranges with the lower bound being inclusive and
        the upper bound being exclusive.
    """
    # Assumes nums is sorted and unique
    # Taken from https://stackoverflow.com/a/48106843
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return [(s, e + 1) for s, e in zip(edges, edges)]


def merge_dicts(d1: dict, d2: dict) -> dict:
    """
    Recursively merges two dict structures. Assumes that the innermost
    dictionaries have unique keys and thus can be merged without concern for collisions.
    """
    for key, value in d2.items():
        if isinstance(value, dict):
            node = d1[key] if key in d1 else {}
            d1[key] = merge_dicts(node, value)
        else:
            # We're at the innermost level, which has unique keys, so just add them
            if isinstance(value, list):
                d1[key] = value + d1[key] if key in d1 else value
            else:
                d1[key] = value
    return d1


def intersect_dicts(d1: dict, d2: dict) -> dict:
    """
    Recursively intersects two dict structures. Assumes that the innermost
    dictionaries have unique keys and thus can be intersected without concern for collisions.
    """
    common_keys = d1.keys() & d2.keys()

    #  Remove all keys and values from d1 that are not in d2
    for key in set(d1.keys()) - common_keys:
        del d1[key]

    for key in common_keys:
        if isinstance(d1[key], dict) and isinstance(d2[key], dict):
            d1[key] = intersect_dicts(d1[key], d2[key])
        else:
            #  At the innermost level we either have a list of values or a list of ranges
            if isinstance(d1[key], list) and isinstance(d1[key][0], int):
                d1[key] = list(set(d1[key]) & set(d2[key]))
            elif isinstance(d1[key], list) and isinstance(d1[key][0], tuple):
                d1[key] = intervals_to_ranges(
                    ranges_to_intervals(d1[key])
                    & ranges_to_intervals(d2[key])  # type: ignore  # mypy can't handle & operators on Interval objects
                )
            else:
                raise ValueError(f"Unsupported value type {type(d1[key])} for intersection.")
    return d1


def defaultdict_to_dict(ddict: Union[dict, defaultdict]) -> dict[Any, Any]:
    if isinstance(ddict, (defaultdict, dict)):
        ddict = {k: defaultdict_to_dict(v) for k, v in ddict.items()}
    return ddict


def run_async_until_complete(call: Any) -> Any:
    """
    Runs a async coroutine until complete and returns its result
    Args:
        call (Any): The coroutine to run.
    Returns:
        Any: The result of the corountine.
    """
    return asyncio.run(call)


def wait_for_key_in_dict(dictionary: dict, key: str, timeout: float) -> bool:
    """
    Busy waits for a key to appear in a dict or timeout is thrown.
    Args:
        dictionary (dict): The dictionary to check.
        key (str): The key to search for.
        timeout (float): How many seconds to wait.
    Returns:
        bool: Whether the key is in the dictionary after timeout seconds.
    """
    timeout_at = time.time() + timeout

    while key not in dictionary and time.time() <= timeout_at:
        time.sleep(0.5)

    return key in dictionary


def numpy_to_native_type(obj: Any) -> Any:
    """
    Converts numpy types to native python types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {numpy_to_native_type(k): numpy_to_native_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_to_native_type(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(numpy_to_native_type(v) for v in obj)
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def return_with_deepcopy_or_noop(to_return: Union[list, dict], copy: bool) -> Union[list, dict]:
    """
    This method either returns the passed object as is, or makes a deep copy
    of it, and returns that.

    Args:
      to_return: the object to be returned
      copy: whether to copy it or not

    Returns:
      The `to_return` object or a copy of it if `copy` is `True`
    """
    return to_return if not copy else deepcopy(to_return)


def merge_property_dicts(left: dict, right: dict, unique_lists: bool = False) -> dict:
    """
    Merge two dictionaries that contain key-value pairs of the form:
        {
            property_name: [property_value_1, ...],
            ...
        }

    Args:
        left: left property dictionary
        right: right property dictionary
        unique_lists: if True, the per-property merged list does not contain
            duplicate values

    Returns:
        The merged dictionaries. The merge should be side-effect safe.
    """
    new_dict = {}
    intersection = set()

    for k, v in left.items():
        if k in right:
            intersection.add(k)
        else:

            new_dict[k] = v.copy()

    for k, v in right.items():
        if k not in intersection:
            new_dict[k] = v.copy()
        else:
            if unique_lists:
                new_dict[k] = list(set(v + left[k]))
            else:
                new_dict[k] = v + left[k]

    return new_dict


def generate_hashable_search_key(
    property_names: list[str], property_values: list[list[str | int | float]], sort_lists: bool = True
) -> str:
    """
    Generate a string representation of a set of property names and values. By default,
    these should be sorted and aligned.

    Args:
        property_names: a list with the property names
        property_values: a list of lists with the property values
        sort_lists: a boolean, indicating whether to sort the two lists (the property_values relative to property_names)

    Returns:
        A string that can be used in a ChunkerIndex to identify ranges fulfilling a certain property
    """
    zipped = list(zip(property_names, property_values))
    if sort_lists:
        zipped.sort(key=lambda x: x[0])
    return ";".join([f"{x}:{y[0]}" for x, y in zipped])  # Take the first value


def ranges_to_intervals(list_of_ranges: List[Tuple[int, int]]) -> List[P.Interval]:
    """Convert a list of ranges to a list of intervals."""
    return P.Interval(*[P.closed(start, end) for start, end in list_of_ranges])


def intervals_to_ranges(intervals: List[P.Interval]) -> List[Tuple[int, int]]:
    """Convert a list of intervals to a list of ranges."""
    return [(int(interval.lower), int(interval.upper)) for interval in intervals]


def generate_hash_string_from_list(string_list: list[str]) -> int:
    """
    Generate a hash string from a list of strings.

    Args:
        string_list: a list of strings to be hashed

    Returns:
        A hash string
    """
    hash_result = hashlib.blake2b()

    for string in string_list:
        hash_result.update(string.encode())

    return int(hash_result.hexdigest(), 16)


def seed_everything(seed: int) -> None:
    """
    Seed all random number generators for reproducibility.

    Args:
        seed: The seed to be used.
    """
    assert isinstance(seed, int), "Seed must be an integer"

    # Cap the seed to be within 0 and 2**32 - 1
    #  Since numpy only accepts 32-bit seeds
    seed = seed % 2**32

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
