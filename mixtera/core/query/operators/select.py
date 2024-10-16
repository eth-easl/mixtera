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

    def generate_sql(self) -> tuple[str, list[Any]]:
        # TODO(#119): This is really janky SQL generation.
        # We should clean this up with a proper query tree again.
        def process_conditions(conditions: list[Tuple[str, str, Any]]) -> tuple[list, list]:
            clauses = []
            params = []
            for field, op, value in conditions:
                if isinstance(value, list) and len(value) > 1:
                    if op == "==":
                        placeholders = ", ".join(["?" for _ in value])
                        clauses.append(f"array_has_any({field}, [{placeholders}])")
                        params.extend(value)
                    elif op == "!=":
                        placeholders = ", ".join(["?" for _ in value])
                        clauses.append(f"NOT array_has_any({field}, [{placeholders}])")
                        params.extend(value)
                    elif op in [">", "<", ">=", "<="]:
                        sub_clauses = [f"any_value({field}) {op} ?" for _ in value]
                        clauses.append(f"({' OR '.join(sub_clauses)})")
                        params.extend(value)
                    else:
                        logger.warning(f"Unsupported operator {op} for list values")
                else:
                    # To call array-contains on lists of length 1 - minor optimization in the generated sql.
                    value = value[0] if isinstance(value, list) else value
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

        # Multiple conditions are interpreted as "AND"
        clauses, params = process_conditions(self.conditions)
        if clauses:
            or_clauses.append(f"({' AND '.join(clauses)})")
            all_params.extend(params)

        # Nested selects are interpreted as "OR"
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

    def __str__(self) -> str:
        return f"select<>({self.conditions})"

    def insert(self, query_plan: QueryPlan) -> Operator:
        if query_plan.is_empty():
            return self
        existing_select = query_plan.root
        existing_select.children.append(self)
        return existing_select
