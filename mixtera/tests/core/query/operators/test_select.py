import unittest
from unittest.mock import MagicMock

from mixtera.core.query import QueryPlan
from mixtera.core.query.operators.select import Select


class TestSelect(unittest.TestCase):
    def setUp(self):
        self.condition = ("field", "==", "value")
        self.select = Select(self.condition)

    def test_init_with_tuple(self):
        self.assertEqual(self.select.conditions, [self.condition])

    def test_init_with_list(self):
        conditions = [("field1", "==", "value1"), ("field2", ">", 5)]
        select = Select(conditions)
        self.assertEqual(select.conditions, conditions)

    def test_init_with_none(self):
        select = Select(None)
        self.assertEqual(select.conditions, [])

    def test_str(self):
        self.assertEqual(str(self.select), "select<>([('field', '==', 'value')])")

    def test_insert_empty_query_plan(self):
        query_plan = QueryPlan()
        query_plan.is_empty = MagicMock(return_value=True)
        result = self.select.insert(query_plan)
        self.assertEqual(result, self.select)

    def test_insert_non_empty_query_plan(self):
        query_plan = QueryPlan()
        existing_select = Select(("field2", "!=", "value2"))
        query_plan.root = existing_select
        query_plan.is_empty = MagicMock(return_value=False)
        result = self.select.insert(query_plan)
        self.assertEqual(result, existing_select)
        self.assertIn(self.select, existing_select.children)

    def test_generate_sql_single_condition(self):
        sql, params = self.select.generate_sql()
        expected_sql = "SELECT * FROM samples WHERE (array_contains(field, ?))"
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, ["value"])

    def test_generate_sql_multiple_conditions(self):
        select = Select([("field1", "==", "value1"), ("field2", ">", 5)])
        sql, params = select.generate_sql()
        expected_sql = "SELECT * FROM samples WHERE (array_contains(field1, ?) AND any_value(field2) > ?)"
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, ["value1", 5])

    def test_generate_sql_no_conditions(self):
        select = Select(None)
        sql, params = select.generate_sql()
        expected_sql = "SELECT * FROM samples"
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, [])

    def test_generate_sql_with_list_values(self):
        select = Select([("field1", "==", ["value1", "value2"]), ("field2", ">", [5, 10])])
        sql, params = select.generate_sql()
        expected_sql = (
            "SELECT * FROM samples WHERE (array_has_any(field1, [?, ?])"
            + " AND (any_value(field2) > ? OR any_value(field2) > ?))"
        )
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, ["value1", "value2", 5, 10])

    def test_generate_sql_with_nested_select(self):
        child_select = Select(("field2", "!=", "value2"))
        self.select.children.append(child_select)
        sql, params = self.select.generate_sql()
        expected_sql = "SELECT * FROM samples WHERE (array_contains(field, ?)) OR (NOT array_contains(field2, ?))"
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, ["value", "value2"])


if __name__ == "__main__":
    unittest.main()
