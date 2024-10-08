import asyncio
import hashlib
import os
import random
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, List, Union

import numpy as np


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
    #  Step 1: Convert the dictionary to a list of key-value pairs
    items = list(d.items())

    # Step 2: Sort the list of key-value pairs
    items.sort()

    #  Step 3: Convert each value list to a hash
    items = [(k, hash_list(v)) for k, v in items]

    # Step 4: Convert the list of key-value pairs to a hash
    return hash_list([f"{k}:{v}" for k, v in items])


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
