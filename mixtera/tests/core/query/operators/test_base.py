import unittest
from unittest.mock import MagicMock, patch

from mixtera.core.query.operators._base import Operator
from mixtera.core.query.query_plan import QueryPlan


class TestOperator(unittest.TestCase):
    def setUp(self):
        self.operator = Operator()

    def test_init(self):
        self.assertEqual(self.operator.children, [])

    def test_str(self):
        self.assertEqual(str(self.operator), "Operator")

    def test_insert_empty(self):
        query_plan = QueryPlan()
        query_plan.is_empty = MagicMock(return_value=True)
        result = self.operator.insert(query_plan)
        self.assertEqual(result, self.operator)
        self.assertEqual(self.operator.children, [])

    def test_insert_non_empty(self):
        mock_child = MagicMock(spec=Operator)
        query_plan = QueryPlan()
        query_plan.root = mock_child
        query_plan.is_empty = MagicMock(return_value=False)

        result = self.operator.insert(query_plan)
        self.assertEqual(result, self.operator)
        self.assertEqual(self.operator.children, [mock_child])

    def test_display(self):
        child1 = Operator()
        child2 = Operator()
        self.operator.children = [child1, child2]

        with patch("builtins.print") as mock_print:
            self.operator.display(0)
            mock_print.assert_any_call("Operator")
            mock_print.assert_any_call("-> Operator")
            mock_print.assert_any_call("-> Operator")

    def test_string(self):
        child1 = Operator()
        child2 = Operator()
        self.operator.children = [child1, child2]

        expected_string = "Operator\n-> Operator\n-> Operator\n"
        self.assertEqual(self.operator.string(0), expected_string)

    def test_generate_sql_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.operator.generate_sql(schema=None)


if __name__ == "__main__":
    unittest.main()
