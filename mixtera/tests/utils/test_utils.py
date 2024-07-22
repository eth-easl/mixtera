import asyncio
import threading
import time

import numpy as np
import portion as P
import pytest
from mixtera.utils import (
    flatten,
    intersect_dicts,
    intervals_to_ranges,
    merge_dicts,
    numpy_to_native_type,
    ranges,
    ranges_to_intervals,
    run_async_until_complete,
    wait_for_key_in_dict,
)
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


def test_basic_merge():
    assert merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}, "Basic merge failed."


def test_overlapping_keys():
    assert merge_dicts({"a": 1}, {"a": 2}) == {"a": 2}, "Overlapping keys test failed."
    assert merge_dicts({"a": [1, 2]}, {"a": [3, 4]}) == {"a": [3, 4, 1, 2]}, "Overlapping keys with lists test failed."


def test_nested_dictionaries():
    assert merge_dicts({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}, "Nested dictionaries test failed."


def test_deeply_nested_structures():
    assert merge_dicts({"a": {"b": {"c": 1}}}, {"a": {"b": {"d": 2}}}) == {
        "a": {"b": {"c": 1, "d": 2}}
    }, "Deeply nested structures test failed."


def test_empty_dictionaries_merge():
    assert not merge_dicts({}, {}), "Empty dictionaries (both) test failed."
    assert merge_dicts({"a": 1}, {}) == {"a": 1}, "Empty dictionary (second) test failed."
    assert merge_dicts({}, {"b": 2}) == {"b": 2}, "Empty dictionary (first) test failed."


def test_non_dict_values():
    assert merge_dicts({"a": [1, 2]}, {"a": [3, 4]}) == {"a": [3, 4, 1, 2]}, "Non-dict values test failed."


def test_mixed_keys():
    assert merge_dicts({1: "a"}, {"2": "b"}) == {1: "a", "2": "b"}, "Mixed keys test failed."


def test_basic_intersection():
    assert not intersect_dicts({}, {"a": [1]}), "Basic intersection failed."


def test_overlapping_keys_with_simple_values():
    assert intersect_dicts({"a": [1], "b": [2]}, {"b": [2], "c": [3]}) == {
        "b": [2]
    }, "Overlapping keys with simple values failed."


def test_list_of_values_intersection():
    assert intersect_dicts({"a": [1, 2, 3]}, {"a": [2, 3, 4]}) == {"a": [2, 3]}, "List of values intersection failed."


def test_list_of_ranges_intersection():
    # Assuming ranges_to_intervals and intervals_to_ranges are defined and work as expected
    assert intersect_dicts({"a": [(1, 3), (4, 6)]}, {"a": [(2, 5)]}) == {
        "a": [(2, 3), (4, 5)]
    }, "List of ranges intersection failed."


def test_nested_dictionaries_with_values():
    assert intersect_dicts({"a": {"b": [1]}}, {"a": {"b": [1], "c": [2]}}) == {
        "a": {"b": [1]}
    }, "Nested dictionaries with values failed."


def test_nested_dictionaries_with_lists():
    assert intersect_dicts({"a": {"b": [1, 2, 3]}}, {"a": {"b": [2, 3, 4]}}) == {
        "a": {"b": [2, 3]}
    }, "Nested dictionaries with lists failed."


def test_nested_dictionaries_with_ranges():
    # Assuming ranges_to_intervals and intervals_to_ranges are defined and work as expected
    assert intersect_dicts({"a": {"b": [(1, 3), (4, 6)]}}, {"a": {"b": [(2, 5)]}}) == {
        "a": {"b": [(2, 3), (4, 5)]}
    }, "Nested dictionaries with ranges failed."


def test_empty_dictionaries_intersect():
    assert not intersect_dicts({}, {}), "Empty dictionaries test failed."
    assert not intersect_dicts({"a": [1]}, {}), "Empty dictionary (second) test failed."
    assert not intersect_dicts({}, {"b": [2]}), "Empty dictionary (first) test failed."


def test_ranges_to_intervals():
    list_of_ranges = [(1, 3), (5, 7)]
    expected_intervals = P.Interval(P.closed(1, 3), P.closed(5, 7))
    result_intervals = ranges_to_intervals(list_of_ranges)

    assert result_intervals == expected_intervals, "The intervals do not match the expected result."


def test_intervals_to_ranges():
    intervals = P.Interval(P.closed(1, 3), P.closed(5, 7))
    expected_ranges = [(1, 3), (5, 7)]
    result_ranges = intervals_to_ranges(intervals)

    assert result_ranges == expected_ranges, "The ranges do not match the expected result."
