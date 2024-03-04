import asyncio
import time
from collections import defaultdict
from typing import Any, List, Tuple, Union

import numpy as np


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


def merge_defaultdicts(d1: defaultdict, d2: defaultdict) -> defaultdict:
    """
    Recursively merges two defaultdict structures. Assumes that the innermost
    dictionaries have unique keys and thus can be merged without concern for collisions.
    """
    for key, value in d2.items():
        if isinstance(value, defaultdict):
            node = d1[key]
            d1[key] = merge_defaultdicts(node, value)
        else:
            # We're at the innermost level, which has unique keys, so just add them
            d1[key] = value
    return d1


def defaultdict_to_dict(ddict: Union[dict, defaultdict]) -> dict[Any, Any]:
    if isinstance(ddict, (defaultdict, dict)):
        ddict = {k: defaultdict_to_dict(v) for k, v in ddict.items()}
    return ddict


def run_in_async_loop_and_return(call: Any) -> Any:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(call)
    finally:
        loop.close()

    return result


def wait_for_key_in_dict(dictionary: dict, key: str, timeout: float) -> bool:
    # TODO(MaxiBoether): rewrite this better
    timeout_at = time.time() + timeout

    while key not in dictionary and time.time() <= timeout_at:
        time.sleep(0.1)

    return key in dictionary

    # end_time = asyncio.get_event_loop().time() + timeout
    # while True:
    #    if key in dictionary:
    #        return True
    #    if asyncio.get_event_loop().time() >= end_time:
    #        return False
    #    await asyncio.sleep(0.1)


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
