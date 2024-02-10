from mixtera.utils import flatten, ranges


def test_flatten():
    assert flatten([[1, 2, 3, 4]]) == [1, 2, 3, 4]
    assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert flatten([[1, 2], [3, 4], [5, 6]]) == [1, 2, 3, 4, 5, 6]


def test_ranges():
    assert ranges([2, 3, 4, 7, 8, 9, 15]) == [(2, 5), (7, 10), (15, 16)]
    assert ranges(list(range(100))) == [(0, 100)]
    assert ranges([0]) == [(0, 1)]
    assert ranges([]) == []
