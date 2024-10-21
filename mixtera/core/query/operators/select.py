from typing import Any, Tuple, Union

from loguru import logger
from mixtera.core.query.query import QueryPlan

from ._base import Operator

Condition = Union[Tuple[str, str, Any], list[Tuple[str, str, Any]], None]


class Select(Operator):
    def __init__(self, conditions: Condition) -> None:
        super().__init__()
        if isinstance(conditions, tuple):
            self.conditions = [conditions]
        elif isinstance(conditions, list):
            self.conditions = conditions
        else:
            self.conditions = []

    def generate_sql(self, schema: dict) -> tuple[str, list[Any]]:
        # TODO(#119): This is really janky SQL generation.
        # We should clean this up with a proper query tree again.
        where_clause, params = self._generate_conditions(schema)
        if where_clause:
            sql = f"SELECT * FROM samples WHERE {where_clause}"
        else:
            sql = "SELECT * FROM samples"
        return sql, params

    def _generate_conditions(self, schema: dict) -> tuple[str, list[Any]]:
        def process_conditions(conditions: list[Tuple[str, str, Any]]) -> tuple[list[str], list[Any]]:
            clauses = []
            params = []
            for field, op, value in conditions:
                if field not in schema:
                    logger.warning(f"Field '{field}' not found in schema. Skipping condition.")
                    continue
                col_info = schema[field]
                is_array = col_info["multiple"]

                op_map = {"==": "=", "!=": "!=", ">": ">", "<": "<", ">=": ">=", "<=": "<="}
                sql_operator = op_map.get(op)
                if not sql_operator:
                    logger.error(f"Unsupported operator '{op}'. Skipping condition.")
                    continue

                # Handle NULL values
                if value is None:
                    if sql_operator == "=":
                        clauses.append(f"{field} IS NULL")
                    elif sql_operator == "!=":
                        clauses.append(f"{field} IS NOT NULL")
                    else:
                        logger.warning(f"Operator '{op}' is not supported for NULL values.")
                    continue

                # Handle array columns
                if is_array:
                    if isinstance(value, list):
                        if len(value) == 0:
                            logger.error(f"Empty list provided for field '{field}'. Skipping condition.")
                            continue
                        placeholders = ", ".join(["?"] * len(value))
                        if sql_operator == "=":
                            clauses.append(f"array_has_any({field}, [{placeholders}])")
                            params.extend(value)
                        elif sql_operator == "!=":
                            clauses.append(f"NOT array_has_any({field}, [{placeholders}])")
                            params.extend(value)
                        else:
                            # TODO(create issue): For comparison operators on arrays, use any_value for now...
                            # Need to think about that together with numeric values
                            clauses.append(f"any_value({field}) {sql_operator} ?")
                            params.append(value[0])  # Simplification for comparison operators
                    else:
                        if sql_operator == "=":
                            clauses.append(f"array_contains({field}, ?)")
                            params.append(value)
                        elif sql_operator == "!=":
                            clauses.append(f"NOT array_contains({field}, ?)")
                            params.append(value)
                        else:
                            clauses.append(f"any_value({field}) {sql_operator} ?")
                            params.append(value)
                else:
                    # Handle sngle-value columns
                    value = value[0] if isinstance(value, list) and len(value) == 1 else value
                    if isinstance(value, list):
                        if len(value) == 0:
                            logger.warning(f"Empty list provided for field '{field}'. Skipping condition.")
                            continue
                        placeholders = ", ".join(["?"] * len(value))
                        if sql_operator == "=":
                            clauses.append(f"{field} IN ({placeholders})")
                            params.extend(value)
                        elif sql_operator == "!=":
                            clauses.append(f"{field} NOT IN ({placeholders})")
                            params.extend(value)
                        else:
                            sub_clauses = [f"{field} {sql_operator} ?" for _ in value]
                            clauses.append(f"({' OR '.join(sub_clauses)})")
                            params.extend(value)
                    else:
                        clauses.append(f"{field} {sql_operator} ?")
                        params.append(value)
            return clauses, params

        or_clauses = []
        all_params = []

        # Process conditions in this Select operator (AND within the same operator)
        and_clauses, params = process_conditions(self.conditions)
        if and_clauses:
            clause = f"({' AND '.join(and_clauses)})"
            or_clauses.append(clause)
            all_params.extend(params)

        # Process conditions in child Select operators (OR between children)
        for child in self.children:
            if isinstance(child, Select):
                child_clause, child_params = child._generate_conditions(schema)
                if child_clause:
                    or_clauses.append(f"({child_clause})")
                    all_params.extend(child_params)
            else:
                logger.warning(f"Unexpected child type: {type(child)}")

        if or_clauses:
            where_clause = " OR ".join(or_clauses)
        else:
            where_clause = ""

        return where_clause, all_params

    def __str__(self) -> str:
        return f"select<>({self.conditions})"

    def insert(self, query_plan: QueryPlan) -> Operator:
        if query_plan.is_empty():
            return self
        existing_select = query_plan.root
        existing_select.children.append(self)
        return existing_select
