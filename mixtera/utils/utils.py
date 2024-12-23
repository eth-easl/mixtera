import asyncio
import multiprocessing as mp
import os
import pickle
import random
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, List, Optional, Type, Union

import numpy as np
import xxhash
from loguru import logger
from tqdm import tqdm

if TYPE_CHECKING:
    from mixtera.core.datacollection.index import ChunkerIndex


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
    Generate a hash from a list of strings using xxhash.

    Args:
        string_list: A list of strings to be hashed.

    Returns:
        An integer hash value.
    """
    # Sort the list of strings to ensure deterministic hashing
    string_list.sort()

    # Concatenate the strings using a separator unlikely to occur in the strings
    concatenated_string = "\0".join(string_list)

    # Compute the hash using xxhash
    hash_result = xxhash.xxh64_intdigest(concatenated_string)

    return hash_result


def hash_dict(d: dict[str, list[str]]) -> int:
    """
    Generate a hash from a dictionary with string keys and list of strings as values using xxhash.

    Args:
        d: A dictionary to be hashed.

    Returns:
        An integer hash value.
    """
    concatenated_items = []

    # Process each key in sorted order for deterministic hashing
    for key in sorted(d.keys()):
        # Get and sort the list of strings associated with the key
        value_list = d[key]
        value_list.sort()

        # Concatenate the sorted list of strings with a separator
        concatenated_values = "\0".join(value_list)

        # Create a string representation of the key-value pair
        concatenated_item = f"{key}:{concatenated_values}"

        # Add to the list of concatenated items
        concatenated_items.append(concatenated_item)

    # Concatenate all key-value pair strings with a separator
    concatenated_string = "\0".join(concatenated_items)

    # Compute the hash using xxhash
    hash_result = xxhash.xxh64_intdigest(concatenated_string)

    return hash_result


def seed_everything_from_list(seed_list: list[Any]) -> None:
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


# Serialization support
def serialize_file(args: tuple[Path, int, int, list[list[int]]]) -> None:
    mixture_key_dir, dataset_id, file_id, intervals = args
    dataset_dir = mixture_key_dir / f"dataset_{dataset_id}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    intervals_array = np.array(intervals, dtype=np.int64)
    file_path = dataset_dir / f"file_{file_id}.npy"
    np.save(file_path, intervals_array)


def serialize_chunker_index(
    chunker_index: "ChunkerIndex",
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    mixture_keys = list(chunker_index.keys())  # Preserve the original order of keys for reproducibility!
    with open(output_dir / "mixture_keys_order.pkl", "wb") as f:
        pickle.dump(mixture_keys, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save MixtureKey objects and prepare file-level tasks
    file_args_list = []
    for idx, mixture_key in enumerate(tqdm(mixture_keys, desc="Preparing tasks per mixture key.")):
        mixture_key_dir = output_dir / f"mixture_key_{idx}"
        mixture_key_dir.mkdir(parents=True, exist_ok=True)
        # Save the MixtureKey object
        with open(mixture_key_dir / "mixture_key.pkl", "wb") as f:
            pickle.dump(mixture_key, f, protocol=pickle.HIGHEST_PROTOCOL)
        datasets = chunker_index[mixture_key]
        # Prepare tasks for each file
        for dataset_id, files in datasets.items():
            for file_id, intervals in files.items():
                args = (mixture_key_dir, dataset_id, file_id, intervals)
                file_args_list.append(args)

    num_cores = os.cpu_count() or 1
    num_workers = max(num_cores - 4, 1)  # TODO(create issue): Make this configurable.
    num_workers = max(min(num_workers, len(file_args_list)), 1)

    # Use a dummy pool for testing, or a multiprocessing pool otherwise
    in_test = os.environ.get("PYTEST_CURRENT_TEST")
    pool_c = DummyPool if in_test else mp.Pool
    core_string = "" if in_test else f" (using {num_workers} cores)"

    with pool_c(num_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(serialize_file, file_args_list),
                total=len(file_args_list),
                desc=f"Serializing Files{core_string}",
            )
        )


def deserialize_file(args: tuple[Path, int, int, int]) -> tuple[int, int, int, List[List[int]]]:
    file_path, mixture_key_idx, dataset_id, file_id = args
    intervals_array = np.load(file_path)
    intervals = intervals_array.tolist()
    return (mixture_key_idx, dataset_id, file_id, intervals)


def deserialize_chunker_index(input_dir: str | Path) -> "ChunkerIndex":
    input_dir = Path(input_dir)
    with open(input_dir / "mixture_keys_order.pkl", "rb") as f:
        mixture_keys_order = pickle.load(f)

    # Prepare mixture_keys_list and mapping from mixture_key_idx to mixture_key
    mixture_keys_list = []
    mixture_key_dirs = []
    for idx in range(len(mixture_keys_order)):
        mixture_key_dir = input_dir / f"mixture_key_{idx}"
        mixture_key_dirs.append(mixture_key_dir)
        with open(mixture_key_dir / "mixture_key.pkl", "rb") as f:
            mixture_key_loaded = pickle.load(f)
        mixture_keys_list.append(mixture_key_loaded)

    # Prepare file_args_list for all files across all mixture keys
    file_args_list = []

    for mixture_key_idx, mixture_key_dir in enumerate(mixture_key_dirs):
        mixture_key = mixture_keys_list[mixture_key_idx]
        dataset_dirs = [d for d in mixture_key_dir.iterdir() if d.is_dir() and d.name.startswith("dataset_")]
        for dataset_dir in dataset_dirs:
            dataset_id = int(dataset_dir.name.split("_")[1])
            for file_path in dataset_dir.glob("file_*.npy"):
                file_id = int(file_path.stem.split("_")[1])
                args = (file_path, mixture_key_idx, dataset_id, file_id)
                file_args_list.append(args)

    # Now process all files in parallel
    num_cores = os.cpu_count() or 1
    num_workers = max(num_cores - 4, 1)  # TODO(#124): Make this configurable.
    num_workers = max(min(num_workers, len(file_args_list)), 1)

    # Use a dummy pool for testing, or a multiprocessing pool otherwise
    in_test = os.environ.get("PYTEST_CURRENT_TEST")
    pool_c = DummyPool if in_test else mp.Pool
    core_string = "" if in_test else f" (using {num_workers} cores)"

    logger.info(f"Deserializing all files in parallel{core_string}.")

    with pool_c(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(deserialize_file, file_args_list),
                total=len(file_args_list),
                desc=f"Deserializing Files{core_string}",
            )
        )

    # Reconstruct the chunker_index
    chunker_index: dict[Any, Any] = {}
    # Initialize per-mixture key dictionaries
    chunker_index_per_mixture_key: list[dict] = [{} for _ in range(len(mixture_keys_list))]

    for mixture_key_idx, dataset_id, file_id, intervals in tqdm(results, desc="Merging loaded results"):
        mixture_key = mixture_keys_list[mixture_key_idx]
        mixture_key_chunker = chunker_index_per_mixture_key[mixture_key_idx]
        if mixture_key not in mixture_key_chunker:
            mixture_key_chunker[mixture_key] = {}
        datasets = mixture_key_chunker[mixture_key]
        if dataset_id not in datasets:
            datasets[dataset_id] = {}
        datasets[dataset_id][file_id] = intervals

    # Combine chunker_index_per_mixture_key into chunker_index
    for mixture_key_dict in chunker_index_per_mixture_key:
        chunker_index.update(mixture_key_dict)

    return chunker_index


def to_numpy_array(data: Any) -> np.ndarray | None:
    """
    Convert input data to a NumPy array, handling various types like:
    - NumPy arrays
    - Python lists and tuples
    - PyTorch tensors
    - TensorFlow tensors
    - Other array-like structures

    Parameters:
    data: The input data to be converted.

    Returns:
    A NumPy ndarray.

    Raises:
    TypeError: If the data type is not supported for conversion.
    """
    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, (list, tuple)):
        return np.array(data)

    # Check for PyTorch tensor without requiring torch to be installed
    if type(data).__module__ == "torch":
        return data.detach().cpu().numpy()

    # Check for TensorFlow tensor without requiring tensorflow to be installed
    if type(data).__module__.startswith("tensorflow"):
        return data.numpy()

    # Attempt to convert using NumPy's array function
    try:
        return np.array(data)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Unsupported data type ({type(data)}) for conversion to NumPy array:\n{e}")
        return None
