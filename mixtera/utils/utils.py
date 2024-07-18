import asyncio
import hashlib
import multiprocessing as mp
import shutil
import subprocess
import time
from collections import defaultdict
from copy import deepcopy
from functools import cache
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
            d1[key] = value
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


def remove_shm_from_resource_tracker() -> None:
    """
    Monkey-patches multiprocessing.resource_tracker to prevent tracking of SharedMemory.

    This function modifies the behavior of multiprocessing.resource_tracker to ignore
    shared memory resources. It prevents the premature calling of _shmunlink, which can
    cause issues with shared memory management. The problem is that the first process
    to exit would already unlink the shared memory, which is not what we want.

    This is a workaround for a known Python issue: https://bugs.python.org/issue38119

    Note:
        This function should be called before using shared memory in multiprocessing.

    Returns:
        None
    """

    # We need this import, otherwise for some reason also mp.resource_tracker is undefined.
    from multiprocessing import resource_tracker  # noqa: F401 # pylint:disable=import-outside-toplevel,unused-import

    def fix_register(name: str, rtype: str) -> None:
        if rtype == "shared_memory":
            return
        mp.resource_tracker._resource_tracker.register(name, rtype)

    mp.resource_tracker.register = fix_register  # type: ignore

    def fix_unregister(name: str, rtype: str) -> None:
        if rtype == "shared_memory":
            return
        mp.resource_tracker._resource_tracker.unregister(name, rtype)

    mp.resource_tracker.unregister = fix_unregister  # type: ignore

    if "shared_memory" in mp.resource_tracker._CLEANUP_FUNCS:  # type: ignore
        del mp.resource_tracker._CLEANUP_FUNCS["shared_memory"]  # type: ignore


def list_shared_memory() -> str:
    """
    Lists all shared memory segments on the system.

    This function uses the 'ipcs' command to retrieve information about
    shared memory segments currently in use on the system.

    Returns:
        str: A string containing the output of the 'ipcs -m' command,
             which lists shared memory segments.
    """

    try:
        result = subprocess.run(["ipcs", "-m"], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Failed to run command: {e}"


def shm_usage() -> tuple[float, float, float]:
    """
    Retrieves the usage statistics of the shared memory (/dev/shm) filesystem.

    This function calculates the total, used, and free space in the /dev/shm
    filesystem, which is typically used for shared memory on Linux systems.
    The sizes are converted from bytes to megabytes for easier readability.

    Returns:
        tuple[float, float, float]: A tuple containing three float values:
            - total_mb: Total size of /dev/shm in megabytes.
            - used_mb: Used space in /dev/shm in megabytes.
            - free_mb: Free space in /dev/shm in megabytes.

    Note:
        If the /dev/shm filesystem is not found (which may occur on non-Linux
        systems), the function returns (-1, -1, -1) to indicate an error condition.

    """
    try:
        total, used, free = shutil.disk_usage("/dev/shm")

        total_mb = total / (1024 * 1024)
        used_mb = used / (1024 * 1024)
        free_mb = free / (1024 * 1024)

        return total_mb, used_mb, free_mb
    except FileNotFoundError:
        return -1, -1, -1


def hash_string(input_string: str, length: int) -> str:
    """
    Generate a fixed-length hash of a given string using SHA-256.

    This function takes an input string and produces a hexadecimal hash digest
    of the specified length. It uses the SHA-256 hashing algorithm to ensure
    a uniform distribution of hash values.

    Args:
        input_string (str): The string to be hashed.
        length (int): The desired length of the output hash string.
                      Must be less than or equal to 64 (the maximum
                      length of a SHA-256 hexadecimal digest).

    Returns:
        str: A hexadecimal string representation of the hash,
             truncated or padded to the specified length.

    Raises:
        ValueError: If the requested length exceeds 64 characters
                    (the maximum length of a SHA-256 hexadecimal digest).

    Note:
        - The function uses UTF-8 encoding to convert the input string to bytes.
        - If the requested length is shorter than the full SHA-256 digest,
          the digest will be truncated.
        - This function is suitable for generating unique identifiers or
          for basic data integrity checks, but should not be used for
          cryptographic purposes that require the full SHA-256 digest.
    """
    if length > hashlib.sha256().digest_size:
        raise ValueError("Requested length exceeds the maximum allowed by SHA-256")
    hash_obj = hashlib.sha256()
    hash_obj.update(input_string.encode("utf-8"))
    hex_digest = hash_obj.hexdigest()
    return hex_digest[:length]


@cache
def max_shm_len() -> int:
    """
    Determine the maximum length of a shared memory name.

    Returns:
        int: The maximum allowed length for a shared memory name
    """
    base_name = "/a"
    for i in range(1, 500):
        try:
            name = base_name + "a" * i
            shm = mp.shared_memory.SharedMemory(name=name, create=True, size=10)
            shm.close()
            shm.unlink()
        except OSError:
            return i - 1
    return i
