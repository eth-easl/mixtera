import unittest

from mixtera.core.query import Operator, QueryPlan
from mixtera.core.query.operators.union import Union


class MockOperator(Operator):
    def __init__(self, name, results: dict):
        super().__init__()
        self.name = name
        self.results = results

    def display(self, level):
        print(" " * level + self.name)

    def execute(self, mdc):
        del mdc


class TestUnion(unittest.TestCase):
    def setUp(self):
        self.query_a = QueryPlan()
        self.query_b = QueryPlan()
        # Adjusted to mimic the expected nested dictionary structure
        self.query_a.root = MockOperator("query_a", {"property1": {"feature1": {"dataset1": {"file1": [1, 2, 3]}}}})
        self.query_b.root = MockOperator("query_b", {"property1": {"feature1": {"dataset1": {"file1": [2, 3, 4]}}}})
        self.union = Union(self.query_a)

    def test_init(self):
        self.query_a.display()
        self.assertEqual(len(self.union.children), 1)
        self.assertEqual(self.union.children[0], self.query_a.root)

    def test_execute(self):
        self.union.children.append(self.query_b.root)
        self.union.execute(None)
        expected_results = {"property1": {"feature1": {"dataset1": {"file1": [1, 2, 3, 4]}}}}
        self.assertEqual(self.union.results, expected_results)

    def test_union_ranges(self):
        range1 = [(1, 5), (10, 15)]
        range2 = [(3, 7), (14, 20)]

        expected_union = [(1, 7), (10, 20)]

        union_operator = Union(self.query_a)

        actual_union = union_operator._union_ranges(range1, range2)

        self.assertEqual(actual_union, expected_union)

    def test_execute_with_incorrect_children(self):
        with self.assertRaises(AssertionError):
            self.union.execute(None)

    def test_repr(self):
        self.assertEqual(str(self.union), "union<>()")


if __name__ == "__main__":
    unittest.main()
