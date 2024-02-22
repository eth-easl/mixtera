from collections import defaultdict
from typing import Any, List, Tuple, Union
import numpy as np

def flatten(non_flat_list: List[List[Any]]) -> List[Any]:
    return [item for sublist in non_flat_list for item in sublist]


def ranges(nums: List[int]) -> List[Tuple[int, int]]:
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
    if isinstance(ddict, defaultdict):
        ddict = {k: defaultdict_to_dict(v) for k, v in ddict.items()}
    return ddict


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
