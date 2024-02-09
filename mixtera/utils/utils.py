from typing import Any, List, Tuple


def flatten(non_flat_list: List[List[Any]]) -> List[Any]:
    return [item for sublist in non_flat_list for item in sublist]


def ranges(nums: List[int]) -> List[Tuple[int, int]]:
    # Assumes nums is sorted and unique
    # Taken from https://stackoverflow.com/a/48106843
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return [(s, e + 1) for s, e in zip(edges, edges)]

def dict_into_dict(target_index: dict[str, list[Any]], new_index: dict[str, list[Any]]) -> None:
    for index_field, buckets in new_index.items():
        for bucket_key, bucket_vals in buckets.items():
            target_index[index_field][bucket_key].extend(bucket_vals)

def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d
