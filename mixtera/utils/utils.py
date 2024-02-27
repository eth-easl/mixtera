import asyncio
from collections import defaultdict
from typing import Any, List, Tuple, Union


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


def run_in_async_loop_and_return(call: Any) -> Any:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(call)
    finally:
        loop.close()

    return result


async def wait_for_key_in_dict(dictionary: dict, key: str, timeout: float) -> bool:
    end_time = asyncio.get_event_loop().time() + timeout
    while True:
        if key in dictionary:
            return True
        if asyncio.get_event_loop().time() >= end_time:
            return False
        await asyncio.sleep(0.1)
