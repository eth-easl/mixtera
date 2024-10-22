import asyncio
import hashlib
import multiprocessing as mp
import os
import pickle
import random
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Type, Union

import numpy as np
from loguru import logger
from tqdm import tqdm


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

    def imap_unordered(self, func: Callable[[Any], Any], iterable: Iterable[Any]) -> Iterator[Any]:
        logger.info("DummyPool executing functions sequentially with imap_unordered.")
        results = []
        for item in iterable:
            result = func(item)
            results.append(result)
        # Shuffle to simulate unordered results.
        random.shuffle(results)
        yield from results


# Serializaiton


def serialize_mixture_key(args):
    mixture_key, datasets, mixture_key_dir = args
    # Save the MixtureKey object using pickle
    with open(mixture_key_dir / "mixture_key.pkl", "wb") as f:
        pickle.dump(mixture_key, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Serialize datasets
    for dataset_id, files in datasets.items():
        dataset_dir = mixture_key_dir / f"dataset_{dataset_id}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for file_id, intervals in files.items():
            intervals_array = np.array(intervals, dtype=np.int64)
            file_path = dataset_dir / f"file_{file_id}.npy"
            np.save(file_path, intervals_array)


def deserialize_mixture_key(args):
    mixture_key_dir = args
    # Load the MixtureKey object using pickle
    with open(mixture_key_dir / "mixture_key.pkl", "rb") as f:
        mixture_key = pickle.load(f)

    datasets = {}
    for dataset_dir in mixture_key_dir.iterdir():
        if dataset_dir.is_dir() and dataset_dir.name.startswith("dataset_"):
            dataset_id = int(dataset_dir.name.split("_")[1])
            files = {}
            for file_path in dataset_dir.glob("file_*.npy"):
                file_id = int(file_path.stem.split("_")[1])
                intervals_array = np.load(file_path)
                intervals = intervals_array.tolist()
                files[file_id] = intervals
            datasets[dataset_id] = files
    return mixture_key, datasets


def serialize_chunker_index(chunker_index, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    args_list = []
    for idx, (mixture_key, datasets) in enumerate(chunker_index.items()):
        mixture_key_dir = output_dir / f"mixture_key_{idx}"
        mixture_key_dir.mkdir(parents=True, exist_ok=True)
        args_list.append((mixture_key, datasets, mixture_key_dir))

    num_cores = os.cpu_count() or 1
    num_workers = max(num_cores - 4, 1)  # TODO(create issue): Make this configurable.
    num_workers = max(min(num_workers, len(args_list)),1)

    # Use a dummy pool for testing, or a multiprocessing pool otherwise
    in_test = os.environ.get("PYTEST_CURRENT_TEST")
    pool_c = DummyPool if in_test else mp.Pool
    core_string = "" if in_test else f" (using {num_workers} cores)"

    with pool_c(num_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(serialize_mixture_key, args_list),
                total=len(args_list),
                desc=f"Serializing Mixture Keys­{core_string}",
            )
        )


def deserialize_chunker_index(input_dir):
    chunker_index = {}
    input_dir = Path(input_dir)
    mixture_key_dirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("mixture_key_")]
    args_list = mixture_key_dirs

    num_cores = os.cpu_count() or 1
    num_workers = max(num_cores - 4, 1)  # TODO(create issue): Make this configurable.
    num_workers = max(min(num_workers, len(args_list)),1)

    # Use a dummy pool for testing, or a multiprocessing pool otherwise
    in_test = os.environ.get("PYTEST_CURRENT_TEST")
    pool_c = DummyPool if in_test else mp.Pool
    core_string = "" if in_test else f" (using {num_workers} cores)"

    with pool_c(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(deserialize_mixture_key, args_list),
                total=len(args_list),
                desc=f"Deserializing Mixture Keys­{core_string}",
            )
        )

    # Reconstruct chunker_index
    for mixture_key, datasets in results:
        chunker_index[mixture_key] = datasets

    return chunker_index
