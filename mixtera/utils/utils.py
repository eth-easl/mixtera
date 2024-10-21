import asyncio
import hashlib
import os
import random
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional, Type, Union

import numpy as np
from loguru import logger


def flatten(non_flat_list: List[List[Any]]) -> List[Any]:
    return [item for sublist in non_flat_list for item in sublist]


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


def hash_list(string_list: list[str]) -> int:
    """
    Generate hash from a list of strings.

    Args:
        string_list: a list of strings to be hashed

    Returns:
        A hash
    """
    # Sort the list of strings to ensure that the hash is deterministic
    string_list.sort()

    hash_result = hashlib.blake2b()

    for string in string_list:
        hash_result.update(string.encode())

    return int(hash_result.hexdigest(), 16)


def hash_dict(d: dict) -> int:
    """
    Generate a hash from a dictionary.

    Args:
        d: a dictionary to be hashed

    Returns:
        A hash
    """
    return hash(frozenset((k, tuple(sorted(v))) for k, v in d.items()))
    #combined_string = ";".join(f"{k}:{','.join(map(str, sorted(v)))}" for k, v in sorted(d.items()))
    #return hash(combined_string)


def seed_everything_from_list(seed_list: List[Any]) -> None:
    """
    Generate a seed from a list of integers.

    Args:
        seed_list: a list of integers

    Returns:
        A seed
    """
    seed_everything(hash_list([str(x) for x in seed_list]))


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


def is_on_github_actions() -> bool:
    # https://docs.github.com/en/actions/learn-github-actions/variables
    if "CI" in os.environ or os.environ["CI"] or "GITHUB_RUN_ID" in os.environ:
        return True

    return False


def merge_sorted_lists(
    sorted_list1: list[tuple[int, ...]], sorted_list2: list[tuple[int, ...]]
) -> list[tuple[int, ...]]:
    """
    Merges two sorted lists of tuples into a single sorted list of tuples.
    The lists are sorted based on the first element of each tuple.

    Args:
        sorted_list1: A list of tuples, each sorted by the first element.
        sorted_list2: Another list of tuples, each sorted by the first element.

    Returns:
        A merged list of tuples, sorted by the first element of each tuple.
    """
    merged_list = []
    i, j = 0, 0

    while i < len(sorted_list1) and j < len(sorted_list2):
        if sorted_list1[i][0] <= sorted_list2[j][0]:
            merged_list.append(sorted_list1[i])
            i += 1
        else:
            merged_list.append(sorted_list2[j])
            j += 1

    if i < len(sorted_list1):
        merged_list.extend(sorted_list1[i:])

    if j < len(sorted_list2):
        merged_list.extend(sorted_list2[j:])

    return merged_list


def numpy_to_native(value: Any) -> Any:
    if isinstance(value, list):
        return [numpy_to_native(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()

    return value  # Assume it's already a native type


class DummyPool:
    def __init__(self, num_workers: int) -> None:
        del num_workers

    def __enter__(self) -> "DummyPool":
        logger.info("Entering DummyPool.")
        return self

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[Any]
    ) -> None:
        pass

    def map(self, func: Callable[[Any], Any], iterable: Iterable[Any]) -> List[Any]:
        logger.info("DummyPool executing functions sequentially.")
        return list(map(func, iterable))
