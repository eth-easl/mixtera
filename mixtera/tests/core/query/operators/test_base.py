import unittest
from unittest.mock import MagicMock, patch

from mixtera.core.query.operators._base import MixteraDataCollection, Operator
from mixtera.core.query.query import QueryPlan


class TestOperator(unittest.TestCase):
    def setUp(self):
        self.operator = Operator()

    def test_init(self):
        self.assertEqual(self.operator.children, [])
        self.assertEqual(self.operator.results, [])
        self.assertIsNone(self.operator.mdc)

    def test_repr(self):
        self.assertEqual(str(self.operator), "Operator")

    def test_set_datacollection(self):
        mock_data_collection = MagicMock(spec=MixteraDataCollection)
        self.operator.datacollection = mock_data_collection
        self.assertEqual(self.operator.mdc, mock_data_collection)

    def test_insert_empty(self):
        mock_operator = MagicMock(spec=Operator)
        query_plan = QueryPlan()
        query_plan.add(mock_operator)
        self.assertEqual(mock_operator, query_plan.root)

    def test_display(self):
        mock_operator = MagicMock(spec=Operator)
        query_plan = MagicMock(spec=QueryPlan)
        query_plan.is_empty.return_value = False
        query_plan.root = mock_operator
        with patch("builtins.print") as mocked_print:
            self.operator.display(1)
            mocked_print.assert_called_with("-> Operator")

    def test_post_order_traverse(self):
        mock_operator = Operator()
        # execute should be called only once
        with patch.object(Operator, "execute") as mocked_execute:
            mock_operator.post_order_traverse()
            mocked_execute.assert_called_once()

    def test_execute_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.operator.execute()
