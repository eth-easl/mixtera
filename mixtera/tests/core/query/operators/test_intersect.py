import unittest

from mixtera.core.query import Operator, QueryPlan
from mixtera.core.query.operators.intersect import Intersection


class MockOperator(Operator):
    def __init__(self, name, results: list):
        super().__init__()
        self.name = name
        self.results = results

    def display(self, level):
        print(" " * level + self.name)

    def execute(self, ldc):
        del ldc


class TestIntersection(unittest.TestCase):
    def setUp(self):
        self.query_a = QueryPlan()
        self.query_b = QueryPlan()
        self.query_a.root = MockOperator("query_a", [1, 2, 3])
        self.query_b.root = MockOperator("query_b", [2, 3, 4])
        self.intersection = Intersection(self.query_a)

    def test_init(self):
        self.query_a.display()
        self.assertEqual(len(self.intersection.children), 1)
        self.assertEqual(self.intersection.children[0], self.query_a.root)

    def test_execute(self):
        self.intersection.children.append(self.query_b.root)
        self.intersection.execute(None)
        self.assertEqual(self.intersection.results, [2, 3])

    def test_execute_with_incorrect_children(self):
        with self.assertRaises(AssertionError):
            self.intersection.execute(None)

    def test_repr(self):
        self.assertEqual(str(self.intersection), "intersection<>()")


if __name__ == "__main__":
    unittest.main()
