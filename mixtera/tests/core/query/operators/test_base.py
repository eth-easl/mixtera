import unittest
from unittest.mock import MagicMock, patch

from mixtera.core.query.operators._base import MixteraDataCollection, Operator


class TestOperator(unittest.TestCase):
    def setUp(self):
        self.operator = Operator()

    def test_init(self):
        self.assertEqual(self.operator.children, [])
        self.assertEqual(self.operator.results, [])
        self.assertFalse(self.operator._materialized)
        self.assertIsNone(self.operator.mdc)

    def test_repr(self):
        self.assertEqual(repr(self.operator), "Operator")

    def test_set_datacollection(self):
        mock_data_collection = MagicMock(spec=MixteraDataCollection)
        self.operator.set_datacollection(mock_data_collection)
        self.assertEqual(self.operator.mdc, mock_data_collection)

    def test_insert(self):
        mock_operator = MagicMock(spec=Operator)
        self.operator.insert(mock_operator)
        self.assertIn(mock_operator, self.operator.children)

    def test_display(self):
        mock_operator = MagicMock(spec=Operator)
        self.operator.insert(mock_operator)
        with patch("builtins.print") as mocked_print:
            self.operator.display(1)
            mocked_print.assert_called_with("-> Operator")

    def test_post_order_traverse(self):
        mock_operator = MagicMock(spec=Operator)
        self.operator.insert(mock_operator)
        with patch.object(Operator, "apply") as mocked_apply:
            self.operator.post_order_traverse()
            mocked_apply.assert_called_once()

    def test_apply(self):
        with self.assertRaises(NotImplementedError):
            self.operator.apply()
