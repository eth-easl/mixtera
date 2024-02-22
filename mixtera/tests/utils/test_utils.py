from mixtera.utils import flatten, numpy_to_native_type, ranges
import numpy as np

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
