import unittest

from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.query import Operator, Query, QueryPlan


class MockOperator:
    def __init__(self, name):
        self.name = name

    def insert(self, node):
        return self if node is None else node

    def display(self, level):
        print(" " * level + self.name)


class TestQueryPlan(unittest.TestCase):
    def setUp(self):
        self.query_plan = QueryPlan()

    def test_init(self):
        self.assertIsNone(self.query_plan.root)

    def test_add(self):
        operator = MockOperator("test_operator")
        self.query_plan.add(operator)
        self.assertEqual(self.query_plan.root, operator)

    def test_display(self):
        operator = MockOperator("test_operator")
        self.query_plan.add(operator)
        with unittest.mock.patch("builtins.print") as mock_print:
            self.query_plan.display()
            mock_print.assert_called_once_with("test_operator")


class TestQuery(unittest.TestCase):
    def setUp(self):
        self.mdc = MixteraDataCollection.from_directory(".")
        self.query = Query(self.mdc)

    def test_init(self):
        self.assertEqual(self.query.mdc, self.mdc)
        self.assertIsInstance(self.query.query_plan, QueryPlan)

    def test_register(self):
        class TestOperator(Operator):
            def apply(self) -> None:
                self.results = ["test"]

        Query.register(TestOperator)
        self.assertTrue(hasattr(Query, "testoperator"))

    def test_from_datacollection(self):
        query = Query.from_datacollection(self.mdc)
        self.assertEqual(query.mdc, self.mdc)
        self.assertIsInstance(query.query_plan, QueryPlan)

    def test_root(self):
        operator = MockOperator("test_operator")
        self.query.query_plan.add(operator)
        self.assertEqual(self.query.root, operator)

    def test_display(self):
        operator = MockOperator("test_operator")
        self.query.query_plan.add(operator)
        with unittest.mock.patch("builtins.print") as mock_print:
            self.query.display()
            mock_print.assert_called_once_with("test_operator")

    def test_execute(self):
        class TestOperator(Operator):
            def apply(self) -> None:
                self.results = ["test"]

        Query.register(TestOperator)
        query = Query.from_datacollection(self.mdc)
        query.testoperator()
        results = query.execute()
        self.assertEqual(results, ["test"])
