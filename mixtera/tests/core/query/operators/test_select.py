import unittest
from unittest.mock import MagicMock

from mixtera.core.query import QueryPlan
from mixtera.core.query.operators.select import Select


class TestSelect(unittest.TestCase):
    def setUp(self):
        self.condition = ("field", "==", "value")
        self.select = Select(self.condition)
        # Mock schema for testing
        self.schema = {
            "field": {"type": "TEXT", "multiple": False, "nullable": True},
            "field1": {"type": "TEXT", "multiple": False, "nullable": True},
            "field2": {"type": "INTEGER", "multiple": False, "nullable": True},
            "array_field": {"type": "TEXT", "multiple": True, "nullable": True},
        }

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
        sql, params = self.select.generate_sql(self.schema)
        expected_sql = "SELECT * FROM samples WHERE (field = ?)"
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, ["value"])

    def test_generate_sql_multiple_conditions(self):
        select = Select([("field1", "==", "value1"), ("field2", ">", 5)])
        sql, params = select.generate_sql(self.schema)
        expected_sql = "SELECT * FROM samples WHERE (field1 = ? AND field2 > ?)"
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, ["value1", 5])

    def test_generate_sql_no_conditions(self):
        select = Select(None)
        sql, params = select.generate_sql(self.schema)
        expected_sql = "SELECT * FROM samples"
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, [])

    def test_generate_sql_with_list_values(self):
        select = Select([("field1", "==", ["value1", "value2"]), ("field2", ">", [5, 10])])
        sql, params = select.generate_sql(self.schema)
        expected_sql = "SELECT * FROM samples WHERE (field1 IN (?, ?) AND (field2 > ? OR field2 > ?))"
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, ["value1", "value2", 5, 10])

    def test_generate_sql_with_nested_select(self):
        child_select = Select(("field2", "!=", "value2"))
        self.select.children.append(child_select)
        sql, params = self.select.generate_sql(self.schema)
        expected_sql = "SELECT * FROM samples WHERE (field = ?) OR ((field2 != ?))"
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, ["value", "value2"])

    def test_generate_sql_field_not_in_schema(self):
        """Test condition with a field not present in the schema."""
        select = Select([("unknown_field", "==", "value")])
        sql, params = select.generate_sql(self.schema)
        expected_sql = "SELECT * FROM samples"
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, [])

    def test_generate_sql_unsupported_operator(self):
        """Test condition with an unsupported operator."""
        select = Select([("field1", "unsupported_op", "value")])
        sql, params = select.generate_sql(self.schema)
        expected_sql = "SELECT * FROM samples"
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, [])

    def test_generate_sql_with_none_value(self):
        """Test conditions where the value is None."""
        # Test with '=='
        select_eq = Select([("field1", "==", None)])
        sql_eq, params_eq = select_eq.generate_sql(self.schema)
        expected_sql_eq = "SELECT * FROM samples WHERE (field1 IS NULL)"
        self.assertEqual(sql_eq, expected_sql_eq)
        self.assertEqual(params_eq, [])

        # Test with '!='
        select_neq = Select([("field1", "!=", None)])
        sql_neq, params_neq = select_neq.generate_sql(self.schema)
        expected_sql_neq = "SELECT * FROM samples WHERE (field1 IS NOT NULL)"
        self.assertEqual(sql_neq, expected_sql_neq)
        self.assertEqual(params_neq, [])

        # Test with unsupported operator for None
        select_gt = Select([("field1", ">", None)])
        sql_gt, params_gt = select_gt.generate_sql(self.schema)
        expected_sql_gt = "SELECT * FROM samples"
        self.assertEqual(sql_gt, expected_sql_gt)
        self.assertEqual(params_gt, [])

    def test_generate_sql_array_field_equal_single_value(self):
        """Test array field with '==' operator and a single value."""
        select = Select([("array_field", "==", "value1")])
        sql, params = select.generate_sql(self.schema)
        expected_sql = "SELECT * FROM samples WHERE (array_contains(array_field, ?))"
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, ["value1"])

    def test_generate_sql_array_field_equal_list_values(self):
        """Test array field with '==' operator and a list of values."""
        select = Select([("array_field", "==", ["value1", "value2"])])
        sql, params = select.generate_sql(self.schema)
        expected_sql = "SELECT * FROM samples WHERE (array_has_any(array_field, [?, ?]))"
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, ["value1", "value2"])

    def test_generate_sql_array_field_not_equal_single_value(self):
        """Test array field with '!=' operator and a single value."""
        select = Select([("array_field", "!=", "value1")])
        sql, params = select.generate_sql(self.schema)
        expected_sql = "SELECT * FROM samples WHERE (NOT array_contains(array_field, ?))"
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, ["value1"])

    def test_generate_sql_array_field_not_equal_list_values(self):
        """Test array field with '!=' operator and a list of values."""
        select = Select([("array_field", "!=", ["value1", "value2"])])
        sql, params = select.generate_sql(self.schema)
        expected_sql = "SELECT * FROM samples WHERE (NOT array_has_any(array_field, [?, ?]))"
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, ["value1", "value2"])

    def test_generate_sql_array_field_comparison_operator(self):
        """Test array field with comparison operator."""
        select = Select([("array_field", ">", 5)])
        sql, params = select.generate_sql(self.schema)
        expected_sql = "SELECT * FROM samples WHERE (any_value(array_field) > ?)"
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, [5])

    def test_generate_sql_array_field_with_list_value_comparison(self):
        """Test array field with comparison operator and list value."""
        select = Select([("array_field", ">", [5, 10])])
        sql, params = select.generate_sql(self.schema)
        # For comparison operators, only the first value is used
        expected_sql = "SELECT * FROM samples WHERE (any_value(array_field) > ?)"
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, [5])

    def test_generate_sql_empty_value_list(self):
        """Test conditions where the value is an empty list."""
        # For arrays
        select_array = Select([("array_field", "==", [])])
        sql_array, params_array = select_array.generate_sql(self.schema)
        expected_sql_array = "SELECT * FROM samples"
        self.assertEqual(sql_array, expected_sql_array)
        self.assertEqual(params_array, [])

        # For non-arrays
        select_non_array = Select([("field1", "==", [])])
        sql_non_array, params_non_array = select_non_array.generate_sql(self.schema)
        expected_sql_non_array = "SELECT * FROM samples"
        self.assertEqual(sql_non_array, expected_sql_non_array)
        self.assertEqual(params_non_array, [])

    def test_generate_sql_multiple_nested_selects(self):
        """Test multiple nested Select operators."""
        select1 = Select([("field1", "==", "value1")])
        select2 = Select([("field2", ">", 5)])
        select3 = Select([("array_field", "!=", ["value1", "value2"])])
        select1.children.append(select2)
        select1.children.append(select3)
        sql, params = select1.generate_sql(self.schema)
        expected_sql = (
            "SELECT * FROM samples WHERE "
            "(field1 = ?) OR ((field2 > ?)) OR ((NOT array_has_any(array_field, [?, ?])))"
        )
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, ["value1", 5, "value1", "value2"])

    def test_generate_sql_combined_array_and_non_array(self):
        """Test a combination of array and non-array fields."""
        select = Select([("field1", "==", "value1"), ("array_field", "==", "value2"), ("field2", "!=", None)])
        sql, params = select.generate_sql(self.schema)
        expected_sql = (
            "SELECT * FROM samples WHERE (field1 = ? AND array_contains(array_field, ?) " + "AND field2 IS NOT NULL)"
        )
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, ["value1", "value2"])

    def test_generate_sql_array_field_invalid_operator(self):
        """Test array field with an unsupported operator."""
        select = Select([("array_field", "unsupported_op", "value1")])
        sql, params = select.generate_sql(self.schema)
        expected_sql = "SELECT * FROM samples"
        self.assertEqual(sql, expected_sql)
        self.assertEqual(params, [])


if __name__ == "__main__":
    unittest.main()
