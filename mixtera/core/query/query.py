from typing import Any

from mixtera.core.datacollection import MixteraDataCollection
from mixtera.core.query.operators._base import Operator
from mixtera.core.query.operators.materialize import Materialize
from mixtera.core.query.operators.select import Select


class QueryPlan:
    """
    QueryPlan is a tree structure that represents the execution plan of a query.
    """

    def __init__(self) -> None:
        self.root = None

    def add(self, node: Operator) -> None:
        self.root = node.insert(self.root)

    def display(self) -> None:
        if self.root:
            self.root.display(0)


class Query:

    def __init__(self, mdc: MixteraDataCollection) -> None:
        self.mdc = mdc
        self.query_plan = QueryPlan()

    @classmethod
    def register(cls, operator: Operator) -> None:
        """
        register is a classmethod that registers operators for the query.
        By default, all built-in operators (under ./operators) are registered.

        Args:
            operator (_type_): _description_
        """
        op_name = operator.__name__.lower()

        def process_op(self, *args: Any, **kwargs: Any) -> "Query":  # type: ignore[no-untyped-def]
            op: Operator = operator(*args, **kwargs)
            op.set_datacollection(self.mdc)
            self.query_plan.add(op)
            return self

        setattr(cls, op_name, process_op)

    @classmethod
    def from_datacollection(cls, mdc: MixteraDataCollection) -> "Query":
        return cls(mdc)

    @property
    def root(self) -> Operator:
        return self.query_plan.root

    def display(self) -> None:
        self.query_plan.display()

    def execute(self, materialize: bool = False, streaming: bool = False) -> list:
        if materialize:
            mat_op = Materialize(streaming)
            mat_op.set_datacollection(self.mdc)
            self.query_plan.add(mat_op)
        self.root.post_order_traverse()
        return self.root.results


Query.register(Select)
