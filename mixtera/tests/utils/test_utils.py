import asyncio
import threading
import time
from datasets.distributed import split_dataset_by_node

import numpy as np
import pytest

from mixtera.utils import (
    distribute_by_ratio,
    flatten,
    numpy_to_native_type,
    run_async_until_complete,
    wait_for_key_in_dict,
)


def test_flatten():
    assert flatten([[1, 2, 3, 4]]) == [1, 2, 3, 4]
    assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert flatten([[1, 2], [3, 4], [5, 6]]) == [1, 2, 3, 4, 5, 6]


def test_numpy_to_native_types():
    np_array = np.array([1, 2, 3])
    result = numpy_to_native_type(np_array)
    assert isinstance(result, list)
    assert result == [1, 2, 3]

    np_dict = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
    result = numpy_to_native_type(np_dict)
    assert isinstance(result, dict)
    assert result == {"a": [1, 2, 3], "b": [4, 5, 6]}

    np_list = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    result = numpy_to_native_type(np_list)
    assert isinstance(result, list)
    assert result == [[1, 2, 3], [4, 5, 6]]

    np_tuple = (np.array([1, 2, 3]), np.array([4, 5, 6]))
    result = numpy_to_native_type(np_tuple)
    assert isinstance(result, tuple)
    assert result == ([1, 2, 3], [4, 5, 6])

    obj = "This is not a numpy object"
    result = numpy_to_native_type(obj)
    assert result == obj


def test_run_simple_coroutine():
    async def sample_coroutine(value):
        await asyncio.sleep(0.1)
        return value

    coroutine = sample_coroutine("test_value")
    result = run_async_until_complete(coroutine)
    assert result == "test_value"

    coroutine = sample_coroutine("test_value2")
    result = run_async_until_complete(coroutine)
    assert result == "test_value2"


def test_run_with_exception():
    async def raise_exception():
        raise ValueError("error")

    coroutine = raise_exception()
    with pytest.raises(ValueError):
        run_async_until_complete(coroutine)


def test_key_present_before_timeout():
    # Key is already in the dictionary, should return True immediately
    test_dict = {"test_key": "test_value"}
    result = wait_for_key_in_dict(test_dict, "test_key", 1)
    assert result


def test_key_not_present():
    # Key is not in the dictionary and will not be added, should return False
    test_dict = {}
    result = wait_for_key_in_dict(test_dict, "test_key", 0.5)
    assert not result


def test_key_appears_before_timeout():
    # Key is not in the dictionary but will be added before timeout
    test_dict = {}

    def add_key():
        time.sleep(0.5)
        test_dict["test_key"] = "test_value"

    add_key_thread = threading.Thread(target=add_key)
    add_key_thread.start()

    result = wait_for_key_in_dict(test_dict, "test_key", 1.5)
    assert result
    add_key_thread.join()


def test_timeout():
    # Key does not appear within the timeout period
    test_dict = {}
    start_time = time.time()
    result = wait_for_key_in_dict(test_dict, "test_key", 0.5)
    end_time = time.time()
    assert not result
    assert end_time - start_time >= 0.5, "Timeout did not work correctly"


def test_distribute_by_ratio():
    result = distribute_by_ratio(10, [])
    assert not result, f"Expected empty for empty ratios, got {result}"

    result = distribute_by_ratio(0, [0.5, 0.5])
    assert result == [0, 0], f"Expected [0, 0] for n=0, got {result}"

    # 3. Ratios must sum to 1 => expect a ValueError if not
    try:
        distribute_by_ratio(10, [0.4, 0.4])
        assert False, "Expected ValueError when ratios don't sum to 1"
    except ValueError:
        pass  # test passes

    ratios = [0.2, 0.3, 0.5]
    n = 10
    result = distribute_by_ratio(n, ratios)
    assert sum(result) == n, f"Sum is {sum(result)} instead of {n}"
    assert result == [2, 3, 5], f"Expected [2, 3, 5], got {result}"

    ratios = [0.3333, 0.3333, 0.3334]  # sums to ~ 1.0
    n = 6
    result = distribute_by_ratio(n, ratios)
    assert sum(result) == 6, f"Expected total 6, got {result}"

    ratios = [0.1, 0.2, 0.7]
    n = 10_000
    result = distribute_by_ratio(n, ratios)
    assert sum(result) == n, "Does not sum to 10000"
    assert result == [1000, 2000, 7000], f"Expected [1000, 2000, 7000], got {result}"
