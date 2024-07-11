import asyncio
import threading
import time

import numpy as np
import pytest
from mixtera.utils import flatten, numpy_to_native_type, ranges, run_async_until_complete, wait_for_key_in_dict
from mixtera.utils.utils import generate_hashable_search_key, merge_property_dicts


def test_flatten():
    assert flatten([[1, 2, 3, 4]]) == [1, 2, 3, 4]
    assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert flatten([[1, 2], [3, 4], [5, 6]]) == [1, 2, 3, 4, 5, 6]


def test_ranges():
    assert ranges([2, 3, 4, 7, 8, 9, 15]) == [(2, 5), (7, 10), (15, 16)]
    assert ranges(list(range(100))) == [(0, 100)]
    assert ranges([0]) == [(0, 1)]
    assert ranges([]) == []


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


def test_merge_property_dicts():
    dict_one = {
        "a": [1, 2],
        "b": [2, 3],
    }

    dict_two = {"b": [1, 2], "c": [1], "d": []}

    merged_with_unique = merge_property_dicts(dict_one, dict_two, unique_lists=True)
    merged_with_duplicates = merge_property_dicts(dict_one, dict_two, unique_lists=False)

    assert merged_with_unique == {
        "a": [1, 2],
        "b": [1, 2, 3],
        "c": [1],
        "d": [],
    }, "Merged w/ unique values is incorrect"
    assert merged_with_duplicates == {
        "a": [1, 2],
        "b": [1, 2, 2, 3],
        "c": [1],
        "d": [],
    }, "Merged w/ duplicates is incorrect"


def test_generate_hashable_search_key():
    properties_one = ["c", "b", "a"]
    values_one = [["x", "y", "z"], list(range(3, -1, -1)), ["m", "n"]]

    properties_two = ["b", "a", "c"]
    values_two = [["x", "y", "z"], list(range(3, -1, -1))]

    properties_three = ["c", "b"]
    values_three = [["x", "y", "z"], list(range(3, -1, -1)), ["m", "n"]]

    key_one = generate_hashable_search_key(properties_one, values_one, sort_lists=True)
    key_two = generate_hashable_search_key(properties_two, values_two)
    key_three = generate_hashable_search_key(properties_three, values_three, sort_lists=False)

    assert key_one == "a:m;b:3;c:x", f"Generated key is incorrect; should be 'a:m;b:3;c:x' not {key_one}"
    assert key_two == "a:3;b:x", f"Generated key is incorrect; should be 'a:3;b:x' not {key_two}"
    assert key_three == "c:x;b:3", f"Generated key is incorrect; should be 'a:3;b:x' not {key_two}"
