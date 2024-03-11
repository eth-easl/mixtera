import unittest
from unittest.mock import MagicMock

from mixtera.core.query.operators.materialize import Materialize


class TestMaterialize(unittest.TestCase):
    def setUp(self):
        self.materialize = Materialize()

    def test_init(self):
        self.assertEqual(self.materialize.streaming, False)

    def test_execute_with_one_child(self):
        child = MagicMock()
        child.results = [[(1, 2), (3, 4)]]
        self.materialize.children.append(child)
        mdc_mock = MagicMock()
        mdc_mock.get_samples_from_ranges.return_value = [(1, 2), (3, 4)]
        self.materialize.mdc = mdc_mock
        self.materialize.execute()
        self.assertEqual(self.materialize.results, [[(1, 2), (3, 4)]])

        self.materialize = Materialize()
        child.results = [[(1, 2)], [(3, 4)]]
        self.materialize.children.append(child)
        mdc_mock = MagicMock()
        mdc_mock.get_samples_from_ranges.return_value = [(1, 2), (3, 4)]
        self.materialize.mdc = mdc_mock
        self.materialize.execute()
        self.assertEqual(self.materialize.results, [[(1, 2), (3, 4)], [(1, 2), (3, 4)]])

    def test_execute_with_more_than_one_child(self):
        self.materialize.children.append(MagicMock())
        self.materialize.children.append(MagicMock())
        with self.assertRaises(AssertionError):
            self.materialize.execute()

    def test_repr(self):
        mdc_mock = MagicMock()
        self.materialize.mdc = mdc_mock
        self.assertEqual(str(self.materialize), f"materialize<{mdc_mock}>")
