from typing import Any, Union

from mixtera.core.query.query import QueryPlan
from mixtera.utils import defaultdict_to_dict

from ._base import Operator
from .intersect import Intersection

valid_operators = ["==", ">", "<", ">=", "<=", "!="]


class Condition:
    def __init__(self, condition_tuple: tuple[str, str, str]):
        self.field = condition_tuple[0]
        self.operator = condition_tuple[1]
        self.value = condition_tuple[2]

    def __repr__(self) -> str:
        return f"{self.field} {self.operator} {self.value}"

    def meet(self, x: Any) -> bool:
        if self.operator == "==":
            return x == self.value
        if self.operator == ">":
            return x > self.value
        if self.operator == "<":
            return x < self.value
        if self.operator == ">=":
            return x >= self.value
        if self.operator == "<=":
            return x <= self.value
        if self.operator == "!=":
            return x != self.value
        raise RuntimeError(f"Invalid operator: {self.operator}")


class Select(Operator):
    """Select operator is used to filter data based on a condition.

    Args:
        condition (Union[Condition, Tuple]): The condition to filter the data.
    """

    def __init__(self, condition: Union[Condition, tuple[str, str, str]]) -> None:
        super().__init__()
        if isinstance(condition, Condition):
            self.condition = condition
        elif isinstance(condition, tuple):
            assert len(condition) == 3, "Condition must be a tuple of length 3"
            assert condition[1] in valid_operators, f"Invalid operator: {condition[1]}"
            self.condition = Condition(condition)
        else:
            raise RuntimeError(f"Invalid condition: {condition}, must be a Condition or a tuple of length 3")

    def apply(self) -> None:
        assert len(self.children) == 0, f"Select operator must have 0 children, got {len(self.children)}"
        assert self.mdc is not None, "Select operator must have a MixteraDataCollection"
        index = self.mdc.get_index(self.condition.field)
        index = defaultdict_to_dict(index)
        self.results = [index[x] for i, x in enumerate(index) if self.condition.meet(x)]

    def __repr__(self) -> str:
        return f"select<{self.mdc}>({self.condition})"

    def insert(self, query_plan: "QueryPlan") -> Operator:
        """
        The insertion of select operator is slightly different.
        When we insert a select operator into the query plan, we
        ensure the select operator is the leaf node.
        Args:
            root (Operator): _description_

        Returns:
            Operator: _description_
        """
        if query_plan.is_empty():
            return self

        intersection_op = Intersection(query_plan)
        intersection_op.children.append(self)
        return intersection_op
