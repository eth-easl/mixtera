import unittest

from mixtera.core.query import Operator, QueryPlan
from mixtera.core.query.operators.intersect import Intersection


class MockOperator(Operator):
    def __init__(self, name, results: dict):
        super().__init__()
        self.name = name
        self.results = results

    def display(self, level):
        print(" " * level + self.name)

    def execute(self, mdc):
        del mdc


class TestIntersection(unittest.TestCase):
    def setUp(self):
        self.query_a = QueryPlan()
        self.query_b = QueryPlan()
        # Adjusted to mimic the expected nested dictionary structure
        self.query_a.root = MockOperator("query_a", {"property1": {"feature1": {"dataset1": {"file1": [1, 2, 3]}}}})
        self.query_b.root = MockOperator("query_b", {"property1": {"feature1": {"dataset1": {"file1": [2, 3, 4]}}}})
        self.intersection = Intersection(self.query_a)

    def test_init(self):
        self.query_a.display()
        self.assertEqual(len(self.intersection.children), 1)
        self.assertEqual(self.intersection.children[0], self.query_a.root)

    def test_execute(self):
        self.intersection.children.append(self.query_b.root)
        self.intersection.execute(None)
        expected_results = {"property1": {"feature1": {"dataset1": {"file1": [2, 3]}}}}
        self.assertEqual(self.intersection.results, expected_results)

    def test_intersect_ranges(self):
        range1 = [(1, 5), (10, 15)]
        range2 = [(3, 7), (14, 20)]

        expected_intersection = [(3, 5), (14, 15)]

        intersection_operator = Intersection(self.query_a)

        actual_intersection = intersection_operator._intersect_ranges(range1, range2)

        self.assertEqual(actual_intersection, expected_intersection)

    def test_execute_with_incorrect_children(self):
        with self.assertRaises(AssertionError):
            self.intersection.execute(None)

    def test_repr(self):
        self.assertEqual(str(self.intersection), "intersection<>()")


if __name__ == "__main__":
    unittest.main()
