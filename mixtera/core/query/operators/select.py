from typing import Any, Tuple, Union

from loguru import logger
from mixtera.core.query.query import QueryPlan

from ._base import Operator


class Select(Operator):
    def __init__(self, conditions: Union[Tuple[str, str, Any], list[Tuple[str, str, Any]], None]) -> None:
        super().__init__()
        if isinstance(conditions, tuple):
            self.conditions = [conditions]
        elif isinstance(conditions, list):
            self.conditions = conditions
        else:
            self.conditions = []

    def generate_sql(self, connection) -> tuple[str, list[Any]]:
        def process_conditions(conditions):
            clauses = []
            params = []
            for field, op, value in conditions:
                if isinstance(value, list):
                    if op == "==":
                        placeholders = ", ".join(["?" for _ in value])
                        clauses.append(f"array_contains_any({field}, [{placeholders}])")
                        params.extend(value)
                    elif op == "!=":
                        placeholders = ", ".join(["?" for _ in value])
                        clauses.append(f"NOT array_contains_any({field}, [{placeholders}])")
                        params.extend(value)
                    elif op in [">", "<", ">=", "<="]:
                        sub_clauses = [f"any_value({field}) {op} ?" for _ in value]
                        clauses.append(f"({' OR '.join(sub_clauses)})")
                        params.extend(value)
                    else:
                        logger.warning(f"Unsupported operator {op} for list values")
                else:
                    if op == "==":
                        clauses.append(f"array_contains({field}, ?)")
                    elif op == "!=":
                        clauses.append(f"NOT array_contains({field}, ?)")
                    else:
                        clauses.append(f"any_value({field}) {op} ?")
                    params.append(value)
            return clauses, params

        or_clauses = []
        all_params = []

        # Process conditions from this Select object
        clauses, params = process_conditions(self.conditions)
        if clauses:
            or_clauses.append(f"({' AND '.join(clauses)})")
            all_params.extend(params)

        # Process conditions from child Select objects
        for child in self.children:
            if isinstance(child, Select):
                child_clauses, child_params = process_conditions(child.conditions)
                if child_clauses:
                    or_clauses.append(f"({' AND '.join(child_clauses)})")
                    all_params.extend(child_params)
            else:
                logger.warning(f"Unexpected child type: {type(child)}")

        if or_clauses:
            where_clause = " OR ".join(or_clauses)
            sql = f"SELECT * FROM samples WHERE {where_clause}"
        else:
            sql = "SELECT * FROM samples"
        return sql, all_params

        # Generate the ORDER BY clause dynamically
        order_by_columns = ["dataset_id", "file_id"]

        # Get all other column names
        column_query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'samples'
        AND column_name NOT IN ('dataset_id', 'file_id', 'sample_id')
        ORDER BY column_name
        """

        # Execute the column query to get the list of columns
        columns = connection.execute(column_query).fetchall()
        other_columns = sorted([col[0] for col in columns])

        # Add other columns to the ORDER BY list
        order_by_columns.extend(other_columns)

        # Add sample_id as the last sorting criterion
        order_by_columns.append("sample_id")

        # Construct the final SQL query
        sql += f"\nORDER BY {', '.join(order_by_columns)}"

        return sql, all_params

    def __str__(self) -> str:
        return f"select<>({self.conditions})"

    def insert(self, query_plan: QueryPlan) -> Operator:
        if query_plan.is_empty():
            return self
        existing_select = query_plan.root
        existing_select.children.append(self)
        return existing_select
