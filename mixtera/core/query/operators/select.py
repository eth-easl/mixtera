from typing import Any, Union

from mixtera.core.datacollection.index.index_collection import IndexFactory, IndexTypes
from mixtera.core.query.query import QueryPlan

from ._base import Operator
from .intersect import Intersection

valid_operators = ["==", ">", "<", ">=", "<=", "!="]


class Condition:
    def __init__(self, condition_tuple: tuple[str, str, str]):
        self.field = condition_tuple[0]
        self.operator = condition_tuple[1]
        self.value = condition_tuple[2]

    def __str__(self) -> str:
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

    def execute(self) -> None:
        assert len(self.children) == 0, f"Select operator must have 0 children, got {len(self.children)}"
        assert self.mdc is not None, "Select operator must have a MixteraDataCollection"
        # TODO(#42): In a future PR, we may want to only load the
        # index that meets the condition, instead of loading the entire index
        # and then filter the results.
        # Now there's get_dict_index_by_feature_value, maybe we can extend it
        # to support not only equal, but also >, <, etc.
        if (index := self.mdc.get_index(self.condition.field)) is None:
            self.results = {}
            return
        results = index.get_dict_index_by_feature(self.condition.field)
        # now filter the results based on the condition
        results = {k: v for k, v in results.items() if self.condition.meet(k)}
        # to ensure the results are in the same format as the original index
        self.results = IndexFactory.create_index(IndexTypes.IN_MEMORY_DICT_RANGE)
        self.results._index = {self.condition.field: results}

    def __str__(self) -> str:
        return f"select<{self.mdc}>({self.condition})"

    def insert(self, query_plan: "QueryPlan") -> Operator:
        """
        The insertion of select operator is slightly different.
        When we insert a select operator into the query plan, we
        ensure the select operator is the leaf node.
        Args:
            query_plan (QueryPlan): The query to insert into the current operator.

        Returns:
            Operator: The new root of the query plan.
        """
        if query_plan.is_empty():
            return self
        # If the query plan is not empty, there is another select.
        # We need to merge the results of those two selects.
        intersection_op = Intersection(query_plan)
        intersection_op.children.append(self)
        return intersection_op
