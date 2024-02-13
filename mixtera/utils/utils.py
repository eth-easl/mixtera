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


def dict_into_dict(target_index: dict[str, dict[str, list[Any]]], new_index: dict[str, dict[str, list[Any]]]) -> None:
    for index_field, buckets in new_index.items():
        for bucket_key, bucket_vals in buckets.items():
            target_index[index_field][bucket_key].extend(bucket_vals)


def defaultdict_to_dict(ddict: Union[dict, defaultdict]) -> dict[Any, Any]:
    if isinstance(ddict, defaultdict):
        ddict = {k: defaultdict_to_dict(v) for k, v in ddict.items()}
    return ddict
