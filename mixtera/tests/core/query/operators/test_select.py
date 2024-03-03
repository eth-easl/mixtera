import unittest
from unittest.mock import MagicMock

from mixtera.core.query import Intersection, QueryPlan
from mixtera.core.query.operators.select import Condition, Select


class TestSelect(unittest.TestCase):
    def setUp(self):
        self.condition = Condition(("field", "operator", "value"))
        self.select = Select(self.condition)

    def test_init_with_condition(self):
        self.assertEqual(self.select.condition, self.condition)

    def test_init_with_tuple(self):
        condition_tuple = ("field", "==", "value")
        select = Select(condition_tuple)
        self.assertEqual(select.condition.field, "field")
        self.assertEqual(select.condition.operator, "==")
        self.assertEqual(select.condition.value, "value")

    def test_init_with_invalid_tuple(self):
        condition_tuple = ("field", "invalid_operator", "value")
        with self.assertRaises(AssertionError):
            Select(condition_tuple)

    def test_init_with_invalid_condition(self):
        with self.assertRaises(RuntimeError):
            Select("invalid_condition")

    def test_execute_with_no_children(self):
        self.select.mdc = MagicMock()
        self.select.mdc.get_index.return_value = {"value": "index_value"}
        self.select.condition.meet = MagicMock(return_value=True)
        self.select.execute()
        self.assertEqual(self.select.results, ["index_value"])

    def test_execute_with_one_child(self):
        self.select.mdc = MagicMock()
        self.select.mdc.get_index.return_value = {"value": "index_value"}
        self.select.condition.meet = MagicMock(return_value=True)
        self.select.children = [MagicMock()]
        self.assertEqual(self.select.results, [])

    def test_chaining_select(self):
        query_plan = QueryPlan()
        select_1 = Select(("field_1", "==", "value_1"))
        select_2 = Select(("field_2", "==", "value_2"))
        query_plan.add(select_1)
        query_plan.add(select_2)
        self.assertIsInstance(query_plan.root, Intersection)
        self.assertEqual(len(query_plan.root.children), 2)
        self.assertEqual(query_plan.root.children[0], select_1)
        self.assertEqual(query_plan.root.children[1], select_2)

    def test_repr(self):
        self.select.mdc = "mdc"
        self.assertEqual(repr(self.select), "select<mdc>(field operator value)")


class TestCondition(unittest.TestCase):
    def setUp(self):
        self.condition = Condition(("field", "operator", "value"))

    def test_init(self):
        self.assertEqual(self.condition.field, "field")
        self.assertEqual(self.condition.operator, "operator")
        self.assertEqual(self.condition.value, "value")

    def test_meet_equal(self):
        self.condition.operator = "=="
        self.assertTrue(self.condition.meet("value"))
        self.assertFalse(self.condition.meet("other_value"))

    def test_meet_not_equal(self):
        self.condition.operator = "!="
        self.assertFalse(self.condition.meet("value"))
        self.assertTrue(self.condition.meet("other_value"))

    def test_meet_greater_than(self):
        self.condition.operator = ">"
        self.condition.value = 5
        self.assertTrue(self.condition.meet(6))
        self.assertFalse(self.condition.meet(4))

    def test_meet_less_than(self):
        self.condition.operator = "<"
        self.condition.value = 5
        self.assertTrue(self.condition.meet(4))
        self.assertFalse(self.condition.meet(6))

    def test_meet_invalid_operator(self):
        self.condition.operator = "invalid_operator"
        with self.assertRaises(RuntimeError):
            self.condition.meet({"field": "value"})
